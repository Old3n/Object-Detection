package com.old3n.autopilot

import android.content.Context
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.concurrent.Executors

/** Everything the overlay needs for one processed frame. */
data class FrameResult(
    val image: Mat,
    val risk: FrameRisk,
    val lane: LaneEstimate,
    val signDets: List<Pair<Det, SignReading>>,
    val confirmedSigns: Set<SignReading>,
    val night: Boolean,
    val inferenceMs: Long,
    val cocoNames: List<String>,
    val mode: Pipeline.Mode,
)

/**
 * Same per-frame order as the desktop pipeline:
 * night boost -> COCO detection + tracking -> lane -> collision risk, plus the ITSD sign
 * model with 3-frame confirmation.
 *
 * Speed measures for weak head units:
 * - BALANCED runs both models at 480 px (about 2x faster than 640). On the demo photos
 *   it kept sign recall and missed some distant people; ACCURATE (640) is the desktop setting.
 * - The sign model runs on every 2nd frame. Signs are static, so confirmation is counted
 *   over sign-model frames (3 runs = 6 camera frames).
 * - Frames are processed at most 640 px wide, and the two models share the CPU cores.
 */
class Pipeline(context: Context, val mode: Mode) {
    enum class Mode(val label: String, val suffix: String) {
        BALANCED("Balanced 480", "_480"), ACCURATE("Accurate 640", "")
    }

    companion object {
        val STREET_CLASSES = setOf(0, 1, 2, 3, 5, 7, 9)  // person, bicycle, car, motorcycle, bus, truck, traffic light
        const val CONFIDENCE = 0.35f
        const val MAX_WIDTH = 640
        const val SIGN_EVERY = 2
    }

    private val threads = maxOf(1, Runtime.getRuntime().availableProcessors() / 2)
    private val coco = OnnxYolo(context, "coco${mode.suffix}.onnx", "coco_names.json", threads)
    private val signs = OnnxYolo(context, "signs${mode.suffix}.onnx", "signs_names.json", threads)
    private val tracker = IouTracker()
    private val lane = LaneDetector()
    private val risk = RiskAssessor()
    private val confirmer = SignConfirmer()
    private val pool = Executors.newFixedThreadPool(2)
    private val start = System.nanoTime()
    private var frame = 0L
    private var lastSigns: List<Pair<Det, SignReading>> = emptyList()
    private var confirmed: Set<SignReading> = emptySet()

    var nightBoostEnabled = true
    var signsEnabled = true

    fun reset() {
        tracker.reset(); lane.reset(); risk.reset(); confirmer.reset()
        lastSigns = emptyList(); confirmed = emptySet()
    }

    fun close() = pool.shutdown()

    /** Takes ownership of [input]; the returned image must be released by the caller. */
    fun process(input: Mat): FrameResult {
        var rgb = input
        if (rgb.cols() > MAX_WIDTH) {
            val small = Mat()
            Imgproc.resize(rgb, small, Size(MAX_WIDTH.toDouble(), rgb.rows() * MAX_WIDTH / rgb.cols().toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            rgb.release(); rgb = small
        }
        val night = nightBoostEnabled && NightBoost.isNight(rgb)
        val work = if (night) NightBoost.boost(rgb).also { rgb.release() } else rgb
        val runSigns = signsEnabled && frame++ % SIGN_EVERY == 0L

        val t0 = System.nanoTime()
        val signJob = if (runSigns) pool.submit<List<Det>> { signs.detect(work, CONFIDENCE) } else null
        val dets = coco.detect(work, CONFIDENCE, STREET_CLASSES)
        val signRaw = signJob?.get()
        val ms = (System.nanoTime() - t0) / 1_000_000

        val tracks = tracker.update(dets)
        val laneEst = lane.detect(work)
        val corridor = laneEst.corridor.takeIf { laneEst.status == "DETECTED" || laneEst.status == "TRACKED" }
        val report = risk.update(tracks, work.cols(), work.rows(), (System.nanoTime() - start) / 1e9, corridor)
        if (!signsEnabled) { lastSigns = emptyList(); confirmed = emptySet() }
        if (signRaw != null) {
            lastSigns = signRaw.mapNotNull { d -> RoadSigns.kind(signs.names[d.cls])?.let { d to it } }
            confirmed = confirmer.update(lastSigns.map { it.second }.toSet())
        }
        return FrameResult(work, report, laneEst, lastSigns, confirmed, night, ms, coco.names, mode)
    }
}
