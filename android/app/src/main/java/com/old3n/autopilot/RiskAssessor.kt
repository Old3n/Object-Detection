package com.old3n.autopilot

import kotlin.math.abs
import kotlin.math.ln

/** Port of adas/risk_assessment.py. Advisory only; produces no vehicle command. */
enum class Level(val rank: Int) { CLEAR(0), CAUTION(1), HIGH(2), CRITICAL(3) }

typealias Pt = Pair<Float, Float>

data class ObjectRisk(
    val track: Track,
    val objectType: String,
    val inCorridor: Boolean,
    val ttc: Float?,
    val cutInEta: Float?,
    val level: Level,
    val reason: String,
)

data class FrameRisk(val level: Level, val reason: String, val objects: List<ObjectRisk>, val corridor: List<Pt>)

class RiskAssessor(
    private val pedestrianIds: Set<Int> = setOf(0),
    private val vehicleIds: Set<Int> = setOf(1, 2, 3, 5, 7),
    private val minConfidence: Float = 0.20f,
    private val minSamples: Int = 5,
    private val minHistorySeconds: Double = 0.30,
    private val staleSeconds: Double = 1.0,
    private val criticalTtc: Float = 1.5f,
    private val highTtc: Float = 3.0f,
    private val cautionTtc: Float = 5.0f,
    private val cutInHorizon: Float = 1.0f,
) {
    companion object {
        val DEFAULT_CORRIDOR: List<Pt> = listOf(0.30f to 0.40f, 0.70f to 0.40f, 0.95f to 0.95f, 0.05f to 0.95f)

        fun pointInPolygon(x: Float, y: Float, poly: List<Pt>): Boolean {
            var inside = false
            var j = poly.size - 1
            for (i in poly.indices) {
                val (xi, yi) = poly[i]; val (xj, yj) = poly[j]
                val cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
                if (abs(cross) < 1e-7f && x in minOf(xi, xj) - 1e-6f..maxOf(xi, xj) + 1e-6f &&
                    y in minOf(yi, yj) - 1e-6f..maxOf(yi, yj) + 1e-6f) return true
                if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside
                j = i
            }
            return inside
        }

        fun slope(samples: List<Pair<Double, Double>>): Double? {
            if (samples.size < 2) return null
            val t0 = samples[0].first
            val ts = samples.map { it.first - t0 }; val vs = samples.map { it.second }
            val mt = ts.average(); val mv = vs.average()
            val den = ts.sumOf { (it - mt) * (it - mt) }
            if (den <= 1e-12) return null
            return ts.indices.sumOf { (ts[it] - mt) * (vs[it] - mv) } / den
        }
    }

    private class Sample(val t: Double, val w: Double, val cx: Double)

    private val history = HashMap<Int, ArrayDeque<Sample>>()
    private val lastSeen = HashMap<Int, Double>()

    fun reset() { history.clear(); lastSeen.clear() }

    private fun ttc(s: List<Sample>): Float? {
        if (s.size < minSamples || s.last().t - s.first().t < minHistorySeconds) return null
        val k = slope(s.filter { it.w > 0 }.map { it.t to ln(it.w) }) ?: return null
        if (k <= 0.02) return null
        return (1.0 / k).toFloat()
    }

    fun levelForTtc(t: Float?): Level = when {
        t == null || t > cautionTtc -> Level.CLEAR
        t <= criticalTtc -> Level.CRITICAL
        t <= highTtc -> Level.HIGH
        else -> Level.CAUTION
    }

    /** [corridor] in normalized 0..1 image coordinates; points are clamped so a lane fit
     *  slightly off-frame cannot abort the run (the desktop version raises here). */
    fun update(tracks: List<Track>, width: Int, height: Int, now: Double, corridor: List<Pt>? = null): FrameRisk {
        val poly = (corridor ?: DEFAULT_CORRIDOR).map { it.first.coerceIn(0f, 1f) to it.second.coerceIn(0f, 1f) }
        val results = ArrayList<ObjectRisk>()
        for (tr in tracks) {
            val d = tr.det
            if (d.score < minConfidence || d.width <= 1f || d.height <= 1f) continue
            lastSeen[tr.id]?.let { if (now - it > staleSeconds) history.remove(tr.id) }
            lastSeen[tr.id] = now
            val cx = (d.x1 + d.x2) / 2f / width
            val by = d.y2 / height
            val h = history.getOrPut(tr.id) { ArrayDeque() }
            h.addLast(Sample(now, (d.width / width).toDouble(), cx.toDouble()))
            if (h.size > 30) h.removeFirst()
            val samples = h.toList()
            val inCorridor = pointInPolygon(cx, by, poly)
            val t = ttc(samples)
            val lateral = if (samples.size >= minSamples) slope(samples.map { it.t to it.cx }) else null
            var eta: Float? = null
            if (!inCorridor && lateral != null && abs(lateral) > 0.01) {
                for (step in 1..10) {
                    val e = cutInHorizon * step / 10f
                    if (pointInPolygon(cx + (lateral * e).toFloat(), by, poly)) { eta = e; break }
                }
            }
            val type = when (d.cls) { in pedestrianIds -> "pedestrian"; in vehicleIds -> "vehicle"; else -> "other" }
            var level = if (inCorridor && type != "other") levelForTtc(t) else Level.CLEAR
            var reason = ""
            if (level != Level.CLEAR) {
                reason = "${type.replaceFirstChar { it.uppercase() }} track ${tr.id}: TTC %.1fs".format(t)
            } else if (eta != null && type != "other") {
                level = Level.CAUTION
                reason = "${type.replaceFirstChar { it.uppercase() }} track ${tr.id}: corridor entry in %.1fs".format(eta)
            }
            results += ObjectRisk(tr, type, inCorridor, t, eta, level, reason)
        }
        lastSeen.entries.filter { now - it.value > staleSeconds }.map { it.key }.forEach {
            lastSeen.remove(it); history.remove(it)
        }
        if (results.isEmpty()) return FrameRisk(Level.CLEAR, "No eligible tracked obstacles", results, poly)
        val top = results.maxBy { it.level.rank }
        return FrameRisk(top.level, if (top.level != Level.CLEAR) top.reason else "No elevated collision risk", results, poly)
    }
}
