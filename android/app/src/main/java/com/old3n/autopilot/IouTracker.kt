package com.old3n.autopilot

/** Tracked row in the same order as the Python pipeline: box, track id, confidence, class. */
data class Track(val det: Det, val id: Int)

/**
 * Greedy IoU tracker (same-class matching, highest IoU first).
 * Lighter than the desktop ByteTrack: no Kalman motion model and no low-score
 * second pass, so IDs can switch more often under fast motion or occlusion.
 */
class IouTracker(
    private val matchIou: Float = 0.3f,
    private val maxMissedFrames: Int = 15,
) {
    private class State(var det: Det, val id: Int, var missed: Int = 0)

    private val active = ArrayList<State>()
    private var nextId = 1

    fun reset() { active.clear(); nextId = 1 }

    fun update(dets: List<Det>): List<Track> {
        val pairs = ArrayList<Triple<Float, Int, Int>>()
        for ((ti, t) in active.withIndex()) for ((di, d) in dets.withIndex()) {
            if (t.det.cls != d.cls) continue
            val v = YoloDecoder.iou(t.det, d)
            if (v >= matchIou) pairs += Triple(v, ti, di)
        }
        pairs.sortByDescending { it.first }
        val usedT = HashSet<Int>(); val usedD = HashSet<Int>()
        val out = ArrayList<Track>()
        for ((_, ti, di) in pairs) {
            if (ti in usedT || di in usedD) continue
            usedT += ti; usedD += di
            active[ti].det = dets[di]; active[ti].missed = 0
            out += Track(dets[di], active[ti].id)
        }
        for ((ti, t) in active.withIndex()) if (ti !in usedT) t.missed++
        active.removeAll { it.missed > maxMissedFrames }
        for ((di, d) in dets.withIndex()) if (di !in usedD) {
            val s = State(d, nextId++)
            active += s
            out += Track(d, s.id)
        }
        return out
    }
}
