package com.old3n.autopilot

/** One detection in original-image pixels. */
data class Det(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val score: Float, val cls: Int,
) {
    val width get() = x2 - x1
    val height get() = y2 - y1
}

/** Letterbox mapping from a source image into the square model input. */
data class Letterbox(val scale: Float, val padX: Float, val padY: Float)

/**
 * Decodes raw Ultralytics YOLO output laid out as [4 + classes][anchors]
 * (cx, cy, w, h in model pixels, then one score per class) and applies
 * class-aware NMS. Same layout for the COCO and the ITSD sign checkpoints.
 */
object YoloDecoder {
    fun decode(
        output: FloatArray,
        numClasses: Int,
        anchors: Int,
        box: Letterbox,
        srcWidth: Int,
        srcHeight: Int,
        confidence: Float,
        allowed: Set<Int>? = null,
        iou: Float = 0.7f,
        maxDetections: Int = 100,
    ): List<Det> {
        require(output.size == (4 + numClasses) * anchors) { "Unexpected YOLO output size ${output.size}" }
        val candidates = ArrayList<Det>()
        for (a in 0 until anchors) {
            var best = -1
            var bestScore = confidence
            for (c in 0 until numClasses) {
                if (allowed != null && c !in allowed) continue
                val s = output[(4 + c) * anchors + a]
                if (s >= bestScore) { bestScore = s; best = c }
            }
            if (best < 0) continue
            val cx = output[a]; val cy = output[anchors + a]
            val w = output[2 * anchors + a]; val h = output[3 * anchors + a]
            val x1 = ((cx - w / 2 - box.padX) / box.scale).coerceIn(0f, srcWidth.toFloat())
            val y1 = ((cy - h / 2 - box.padY) / box.scale).coerceIn(0f, srcHeight.toFloat())
            val x2 = ((cx + w / 2 - box.padX) / box.scale).coerceIn(0f, srcWidth.toFloat())
            val y2 = ((cy + h / 2 - box.padY) / box.scale).coerceIn(0f, srcHeight.toFloat())
            if (x2 - x1 > 1f && y2 - y1 > 1f) candidates += Det(x1, y1, x2, y2, bestScore, best)
        }
        return nms(candidates, iou, maxDetections)
    }

    fun nms(dets: List<Det>, iou: Float, max: Int): List<Det> {
        val sorted = dets.sortedByDescending { it.score }
        val kept = ArrayList<Det>()
        for (d in sorted) {
            if (kept.size >= max) break
            if (kept.none { it.cls == d.cls && iou(it, d) > iou }) kept += d
        }
        return kept
    }

    fun iou(a: Det, b: Det): Float {
        val iw = (minOf(a.x2, b.x2) - maxOf(a.x1, b.x1)).coerceAtLeast(0f)
        val ih = (minOf(a.y2, b.y2) - maxOf(a.y1, b.y1)).coerceAtLeast(0f)
        val inter = iw * ih
        val union = a.width * a.height + b.width * b.height - inter
        return if (union <= 0f) 0f else inter / union
    }
}
