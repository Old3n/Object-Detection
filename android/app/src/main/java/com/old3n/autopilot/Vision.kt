package com.old3n.autopilot

import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.roundToInt

/** Port of adas/road_signs.py is_night / boost_night (CLAHE on LAB lightness). */
object NightBoost {
    const val NIGHT_BRIGHTNESS = 60.0

    fun isNight(rgb: Mat): Boolean {
        val m = Core.mean(rgb)
        return (m.`val`[0] + m.`val`[1] + m.`val`[2]) / 3.0 < NIGHT_BRIGHTNESS
    }

    fun boost(rgb: Mat): Mat {
        val lab = Mat(); Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        val ch = ArrayList<Mat>(); Core.split(lab, ch)
        Imgproc.createCLAHE(2.0, Size(8.0, 8.0)).apply(ch[0], ch[0])
        Core.merge(ch, lab)
        val out = Mat(); Imgproc.cvtColor(lab, out, Imgproc.COLOR_Lab2RGB)
        lab.release(); ch.forEach { it.release() }
        return out
    }
}

data class LaneEstimate(val status: String, val confidence: Float, val corridor: List<Pt>?)

/** Port of adas/lane_detection.py (colour/edge cues + Hough, smoothing and hold). */
class LaneDetector(
    private val roiTop: Float = 0.56f,
    private val roiBottom: Float = 0.97f,
    private val smoothing: Float = 0.35f,
    private val maxHold: Int = 8,
) {
    private var previous: Pair<FloatArray, FloatArray>? = null
    private var missing = 0
    private var previousConfidence = 0f

    fun reset() { previous = null; missing = 0; previousConfidence = 0f }

    private fun fit(segs: List<IntArray>, top: Float, bottom: Float): FloatArray? {
        if (segs.isEmpty()) return null
        val xs = ArrayList<Double>(); val ys = ArrayList<Double>(); val ws = ArrayList<Double>()
        for (s in segs) {
            val len = hypot((s[2] - s[0]).toDouble(), (s[3] - s[1]).toDouble())
            xs += s[0].toDouble(); xs += s[2].toDouble(); ys += s[1].toDouble(); ys += s[3].toDouble()
            ws += len; ws += len
        }
        if (ys.max() - ys.min() < 20) return null
        // Weighted least squares x = a*y + b with weights sqrt(len), as np.polyfit(w=sqrt(len)).
        val w = ws  // np.polyfit squares its w=sqrt(len) weights, so the effective weight is len
        val sw = w.sum(); val my = ys.indices.sumOf { w[it] * ys[it] } / sw; val mx = ys.indices.sumOf { w[it] * xs[it] } / sw
        val den = ys.indices.sumOf { w[it] * (ys[it] - my) * (ys[it] - my) }
        if (den <= 1e-9) return null
        val a = ys.indices.sumOf { w[it] * (ys[it] - my) * (xs[it] - mx) } / den
        val b = mx - a * my
        return floatArrayOf((a * top + b).toFloat(), (a * bottom + b).toFloat())
    }

    fun detect(rgb: Mat): LaneEstimate {
        val w = rgb.cols(); val h = rgb.rows()
        val gray = Mat(); Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
        val edges = Mat(); Imgproc.Canny(gray, edges, 55.0, 150.0)
        val hsv = Mat(); Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        val white = Mat(); Core.inRange(hsv, Scalar(0.0, 0.0, 175.0), Scalar(180.0, 75.0, 255.0), white)
        val yellow = Mat(); Core.inRange(hsv, Scalar(12.0, 65.0, 75.0), Scalar(42.0, 255.0, 255.0), yellow)
        Core.bitwise_or(white, yellow, white)
        val marking = Mat(); Imgproc.Canny(white, marking, 40.0, 120.0)
        Core.bitwise_or(edges, marking, edges)
        val top = (roiTop * h).roundToInt(); val bottom = (roiBottom * h).roundToInt()
        val roi = Mat.zeros(edges.size(), edges.type())
        Imgproc.fillPoly(roi, listOf(MatOfPoint(
            Point(0.04 * w, bottom.toDouble()), Point(0.39 * w, top.toDouble()),
            Point(0.61 * w, top.toDouble()), Point(0.96 * w, bottom.toDouble()))), Scalar(255.0))
        Core.bitwise_and(edges, roi, edges)
        val lines = Mat()
        Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180, maxOf(18, (w * 0.018).toInt()),
            maxOf(28, (w * 0.035).toInt()).toDouble(), maxOf(24, (w * 0.08).toInt()).toDouble())
        val left = ArrayList<IntArray>(); val right = ArrayList<IntArray>()
        for (i in 0 until lines.rows()) {
            val l = lines.get(i, 0); val s = intArrayOf(l[0].toInt(), l[1].toInt(), l[2].toInt(), l[3].toInt())
            val dx = s[2] - s[0]; if (dx == 0) continue
            val k = (s[3] - s[1]).toDouble() / dx; val mid = (s[0] + s[2]) * 0.5
            if (abs(k) in 0.30..3.5) {
                if (k < 0 && mid < w * 0.58) left += s else if (k > 0 && mid > w * 0.42) right += s
            }
        }
        listOf(gray, edges, hsv, white, yellow, marking, roi, lines).forEach { it.release() }
        var l = fit(left, top.toFloat(), bottom.toFloat()); var r = fit(right, top.toFloat(), bottom.toFloat())
        var valid = l != null && r != null
        if (valid) {
            val bw = r!![1] - l!![1]; val tw = r[0] - l[0]
            valid = bw in 0.18f * w..0.95f * w && tw in 0.08f * w..0.90f * w && l[0] < r[0] && l[1] < r[1] &&
                minOf(l[0], l[1], r[0], r[1]) >= -0.05f * w && maxOf(l[0], l[1], r[0], r[1]) <= 1.05f * w
        }
        if (valid) {
            previous?.let { (pl, pr) -> l = FloatArray(2) { smoothing * l!![it] + (1 - smoothing) * pl[it] }
                r = FloatArray(2) { smoothing * r!![it] + (1 - smoothing) * pr[it] } }
            previous = l!!.copyOf() to r!!.copyOf(); missing = 0
            previousConfidence = minOf(0.98f, 0.40f + 0.055f * minOf(left.size, 6) + 0.055f * minOf(right.size, 6))
            return LaneEstimate("DETECTED", previousConfidence, corridor(l!!, r!!, w))
        }
        missing++
        val p = previous
        if (p != null && missing <= maxHold) {
            return LaneEstimate("TRACKED", previousConfidence * (1f - missing / (maxHold + 1f)), corridor(p.first, p.second, w))
        }
        previous = null; previousConfidence = 0f
        return LaneEstimate("NOT_DETECTED", 0f, null)
    }

    private fun corridor(l: FloatArray, r: FloatArray, w: Int): List<Pt> =
        listOf(l[0] / w to roiTop, r[0] / w to roiTop, r[1] / w to roiBottom, l[1] / w to roiBottom)
}
