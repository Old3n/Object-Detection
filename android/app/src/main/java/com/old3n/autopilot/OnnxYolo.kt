package com.old3n.autopilot

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONArray
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer

/** Runs an Ultralytics YOLO ONNX export (input 1x3xSxS RGB 0..1, letterboxed with 114). */
class OnnxYolo(context: Context, model: String, namesFile: String, threads: Int = 4) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val names: List<String>
    private val size: Int

    init {
        val bytes = context.assets.open(model).use { it.readBytes() }
        session = try {
            // XNNPACK: optimized ARM NEON kernels; it owns the threads, so ORT's own pool stays at 1.
            env.createSession(bytes, options(threads, xnnpack = true))
        } catch (e: Exception) {
            env.createSession(bytes, options(threads, xnnpack = false))
        }
        val json = JSONArray(context.assets.open(namesFile).bufferedReader().readText())
        names = List(json.length()) { json.getString(it) }
        size = (session.inputInfo.values.first().info as ai.onnxruntime.TensorInfo).shape[2].toInt()
    }

    private fun options(threads: Int, xnnpack: Boolean) = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        if (xnnpack) {
            addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
            setIntraOpNumThreads(1)
        } else {
            setIntraOpNumThreads(threads)
        }
    }

    fun detect(rgb: Mat, confidence: Float, allowed: Set<Int>? = null): List<Det> {
        val w = rgb.cols(); val h = rgb.rows()
        val scale = minOf(size.toFloat() / w, size.toFloat() / h)
        val nw = Math.round(w * scale); val nh = Math.round(h * scale)
        val padX = (size - nw) / 2; val padY = (size - nh) / 2
        val resized = Mat(); Imgproc.resize(rgb, resized, Size(nw.toDouble(), nh.toDouble()))
        val boxed = Mat(); Core.copyMakeBorder(resized, boxed, padY, size - nh - padY, padX, size - nw - padX,
            Core.BORDER_CONSTANT, Scalar(114.0, 114.0, 114.0))
        val f = Mat(); boxed.convertTo(f, CvType.CV_32FC3, 1.0 / 255)
        val hwc = FloatArray(size * size * 3); f.get(0, 0, hwc)
        listOf(resized, boxed, f).forEach { it.release() }
        val chw = FloatArray(hwc.size); val plane = size * size
        for (i in 0 until plane) { chw[i] = hwc[3 * i]; chw[plane + i] = hwc[3 * i + 1]; chw[2 * plane + i] = hwc[3 * i + 2] }
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), longArrayOf(1, 3, size.toLong(), size.toLong())).use { input ->
            session.run(mapOf(session.inputNames.first() to input)).use { out ->
                val t = out[0] as OnnxTensor
                val shape = t.info.shape
                val data = FloatArray(t.floatBuffer.remaining()); t.floatBuffer.get(data)
                return YoloDecoder.decode(data, shape[1].toInt() - 4, shape[2].toInt(),
                    Letterbox(scale, padX.toFloat(), padY.toFloat()), w, h, confidence, allowed)
            }
        }
    }
}
