package com.old3n.autopilot

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp

/** Mirrors tests/test_risk_assessment.py and tests/test_road_signs.py on the Kotlin ports. */
class LogicTest {
    private fun box(cx: Float, w: Float, bottom: Float = 650f, id: Int = 1, cls: Int = 2) =
        Track(Det(cx - w / 2, 200f, cx + w / 2, bottom, 0.9f, cls), id)

    @Test fun approachingVehicleEscalatesWithExpectedTtc() {
        val r = RiskAssessor()
        var rep: FrameRisk? = null
        for (i in 0 until 8) rep = r.update(listOf(box(640f, 100f * exp(0.5f * i * 0.1f))), 1280, 720, i * 0.1)
        assertEquals(Level.HIGH, rep!!.level)
        assertEquals(2.0f, rep.objects[0].ttc!!, 1e-3f)
    }

    @Test fun recedingAndOtherClassesStayClear() {
        val r = RiskAssessor()
        var rep: FrameRisk? = null
        for (i in 0 until 8) rep = r.update(listOf(box(640f, 100f * exp(-0.5f * i * 0.1f)), box(640f, 300f, id = 2, cls = 9)), 1280, 720, i * 0.1)
        assertEquals(Level.CLEAR, rep!!.level)
        assertNull(rep.objects[0].ttc)
    }

    @Test fun lateralMotionPredictsCutIn() {
        val r = RiskAssessor()
        var rep: FrameRisk? = null
        // Corridor right edge at this height is x≈0.927; the object stays outside but drifts in.
        for (i in 0 until 5) rep = r.update(listOf(box(1280f * (0.98f - 0.01f * i), 60f)), 1280, 720, i * 0.1)
        assertEquals(Level.CAUTION, rep!!.level)
        assertNotNull(rep.objects[0].cutInEta)
    }

    @Test fun offFrameLaneCorridorIsClampedNotFatal() {
        val corridor = listOf(-0.03f to 0.56f, 0.55f to 0.56f, 1.02f to 0.97f, -0.04f to 0.97f)
        val rep = RiskAssessor().update(listOf(box(640f, 100f)), 1280, 720, 0.0, corridor)
        assertTrue(rep.corridor.all { it.first in 0f..1f })
    }

    @Test fun signKindsMatchPython() {
        assertEquals(SignReading("stop"), RoadSigns.kind("d-xx-1"))
        assertEquals(SignReading("speed", 30), RoadSigns.kind("d-r-30"))
        assertEquals(SignReading("junction"), RoadSigns.kind("junction_crossroad"))
        listOf("car", "d-r-20", "speed_80", "d-r-33").forEach { assertNull(RoadSigns.kind(it)) }
        assertEquals("STOP | SPEED 30 | SPEED 50",
            RoadSigns.text(setOf(SignReading("speed", 50), SignReading("stop"), SignReading("speed", 30))))
    }

    @Test fun signNeedsThreeConsecutiveFrames() {
        val c = SignConfirmer(); val stop = SignReading("stop")
        assertTrue(c.update(setOf(stop)).isEmpty())
        assertTrue(c.update(setOf(stop)).isEmpty())
        assertEquals(setOf(stop), c.update(setOf(stop, SignReading("speed", 30))))
        assertTrue(c.update(emptySet()).isEmpty())
    }

    @Test fun trackerKeepsIdAcrossFrames() {
        val t = IouTracker()
        val a = t.update(listOf(Det(100f, 100f, 200f, 300f, .9f, 0)))
        val b = t.update(listOf(Det(110f, 105f, 210f, 305f, .9f, 0), Det(600f, 100f, 700f, 300f, .9f, 2)))
        assertEquals(a[0].id, b.first { it.det.cls == 0 }.id)
        assertEquals(2, b.map { it.id }.toSet().size)
    }

    @Test fun decoderUndoesLetterboxAndRunsNms() {
        val anchors = 3; val classes = 2
        val out = FloatArray((4 + classes) * anchors)
        fun put(a: Int, cx: Float, cy: Float, w: Float, h: Float, s0: Float, s1: Float) {
            out[a] = cx; out[anchors + a] = cy; out[2 * anchors + a] = w; out[3 * anchors + a] = h
            out[4 * anchors + a] = s0; out[5 * anchors + a] = s1
        }
        put(0, 320f, 320f, 100f, 100f, 0.9f, 0.1f)
        put(1, 322f, 321f, 100f, 100f, 0.8f, 0.1f)  // duplicate, removed by NMS
        put(2, 100f, 320f, 40f, 40f, 0.1f, 0.2f)    // below confidence
        val dets = YoloDecoder.decode(out, classes, anchors, Letterbox(0.5f, 0f, 140f), 1280, 720, 0.35f)
        assertEquals(1, dets.size)
        assertEquals(540f, dets[0].x1, 1e-3f); assertEquals(260f, dets[0].y1, 1e-3f)
    }
}
