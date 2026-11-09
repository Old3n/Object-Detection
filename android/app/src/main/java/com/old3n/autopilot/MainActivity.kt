package com.old3n.autopilot

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var view: ImageView
    private lateinit var status: TextView
    private var pipeline: Pipeline? = null
    private val analysis = Executors.newSingleThreadExecutor()
    private var mode = Pipeline.Mode.BALANCED
    private var rowBuffer = ByteArray(0)
    private var lastFrameNs = 0L
    private var fps = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        view = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER; setBackgroundColor(0xFF000000.toInt()) }
        status = TextView(this).apply { setTextColor(-1); textSize = 18f; gravity = Gravity.CENTER; text = "Loading models…" }
        val controls = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val night = Button(this).apply { text = "Night boost: AUTO" }
        val signs = Button(this).apply { text = "Signs: ON" }
        val reset = Button(this).apply { text = "Reset tracks" }
        val quality = Button(this).apply { text = mode.label }
        night.setOnClickListener {
            pipeline?.let { it.nightBoostEnabled = !it.nightBoostEnabled; night.text = "Night boost: " + if (it.nightBoostEnabled) "AUTO" else "OFF" }
        }
        signs.setOnClickListener {
            pipeline?.let { it.signsEnabled = !it.signsEnabled; signs.text = "Signs: " + if (it.signsEnabled) "ON" else "OFF" }
        }
        reset.setOnClickListener { pipeline?.reset() }
        quality.setOnClickListener {
            mode = if (mode == Pipeline.Mode.BALANCED) Pipeline.Mode.ACCURATE else Pipeline.Mode.BALANCED
            quality.text = mode.label; status.text = "Loading ${mode.label}…"
            val old = pipeline
            analysis.execute { old?.close(); pipeline = null; loadPipeline(night, signs) }
        }
        listOf(quality, night, signs, reset).forEach { controls.addView(it) }
        setContentView(FrameLayout(this).apply {
            addView(view); addView(status)
            addView(controls, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.END).apply { bottomMargin = 60 })
        })

        if (!OpenCVLoader.initLocal()) { status.text = "OpenCV failed to load"; return }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) start()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
    }

    override fun onRequestPermissionsResult(code: Int, permissions: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, permissions, results)
        if (results.firstOrNull() == PackageManager.PERMISSION_GRANTED) start()
        else status.text = "Camera permission is required for the live test."
    }

    private fun loadPipeline(night: Button? = null, signs: Button? = null) {
        try {
            pipeline = Pipeline(this, mode)
            runOnUiThread {
                status.text = ""
                night?.text = "Night boost: AUTO"; signs?.text = "Signs: ON"
            }
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed", e)
            runOnUiThread { status.text = "Model load failed: ${e.message}" }
        }
    }

    private fun start() {
        analysis.execute {
            try {
                pipeline = Pipeline(this, mode)
                runOnUiThread { status.text = ""; bindCamera() }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { status.text = "Model load failed: ${e.message}" }
            }
        }
    }

    private fun bindCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector = ResolutionSelector.Builder()
                .setResolutionStrategy(ResolutionStrategy(android.util.Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val useCase = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            useCase.setAnalyzer(analysis) { onFrame(it) }
            val camera = if (provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)) CameraSelector.DEFAULT_BACK_CAMERA
            else CameraSelector.DEFAULT_FRONT_CAMERA
            provider.unbindAll()
            provider.bindToLifecycle(this, camera, useCase)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrame(proxy: ImageProxy) {
        val p = pipeline ?: return proxy.close()
        try {
            val rgb = toUprightRgb(proxy)
            val result = p.process(rgb)
            val now = System.nanoTime()
            if (lastFrameNs > 0) { val inst = 1e9f / (now - lastFrameNs); fps = if (fps == 0f) inst else 0.85f * fps + 0.15f * inst }
            lastFrameNs = now
            // A fresh bitmap per frame: the one on screen is never drawn into while displayed.
            val out = Overlay.render(result, fps, null)
            result.image.release()
            runOnUiThread { view.setImageBitmap(out) }
        } catch (e: Exception) {
            Log.e(TAG, "Frame failed", e)
        } finally {
            proxy.close()
        }
    }

    private fun toUprightRgb(proxy: ImageProxy): Mat {
        // Copy the RGBA plane straight into a Mat (no Bitmap round trip); rowStride may include padding.
        val plane = proxy.planes[0]
        val buffer = plane.buffer.also { it.rewind() }
        val size = plane.rowStride * proxy.height
        if (rowBuffer.size < size) rowBuffer = ByteArray(size)
        buffer.get(rowBuffer, 0, minOf(size, buffer.remaining()))
        val rgba = Mat(proxy.height, plane.rowStride / plane.pixelStride, CvType.CV_8UC4)
        rgba.put(0, 0, rowBuffer, 0, size)
        val cropped = rgba.submat(0, proxy.height, 0, proxy.width)
        val rgb = Mat(); Imgproc.cvtColor(cropped, rgb, Imgproc.COLOR_RGBA2RGB)
        rgba.release()
        val code = when (proxy.imageInfo.rotationDegrees) {
            90 -> Core.ROTATE_90_CLOCKWISE; 180 -> Core.ROTATE_180; 270 -> Core.ROTATE_90_COUNTERCLOCKWISE; else -> -1
        }
        if (code < 0) return rgb
        val rotated = Mat(); Core.rotate(rgb, rotated, code); rgb.release()
        return rotated
    }

    override fun onDestroy() { super.onDestroy(); analysis.shutdown() }

    companion object { private const val TAG = "AutoPilot" }
}
