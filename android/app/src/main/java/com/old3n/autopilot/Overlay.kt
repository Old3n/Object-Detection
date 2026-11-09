package com.old3n.autopilot

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import org.opencv.android.Utils

/** Draws the dashboard onto a bitmap, mirroring the desktop RiskVideoBenchmark overlay. */
object Overlay {
    private val RISK_COLORS = mapOf(
        Level.CLEAR to Color.rgb(0, 170, 0), Level.CAUTION to Color.rgb(255, 165, 0),
        Level.HIGH to Color.rgb(255, 100, 0), Level.CRITICAL to Color.rgb(255, 0, 0),
    )
    private val SIGN_COLOR = Color.rgb(255, 140, 0)

    fun render(r: FrameResult, fps: Float, reuse: Bitmap?): Bitmap {
        val w = r.image.cols(); val h = r.image.rows()
        val bmp = if (reuse != null && reuse.width == w && reuse.height == h) reuse
        else Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(r.image, bmp)
        val c = Canvas(bmp)
        val s = maxOf(w, h) / 960f  // frames are ~640 px wide; keep labels readable
        val stroke = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 3f * s; isAntiAlias = true }
        val fill = Paint().apply { style = Paint.Style.FILL }
        val text = Paint().apply { isAntiAlias = true; textSize = 24f * s; color = Color.WHITE; isFakeBoldText = true }

        if (r.lane.corridor != null) {
            val p = Path()
            r.risk.corridor.forEachIndexed { i, (x, y) -> if (i == 0) p.moveTo(x * w, y * h) else p.lineTo(x * w, y * h) }
            p.close()
            fill.color = Color.argb(28, 0, 255, 255); c.drawPath(p, fill)
            stroke.color = if (r.lane.status == "DETECTED") Color.rgb(0, 220, 0) else Color.rgb(255, 190, 0)
            c.drawPath(p, stroke)
        }
        for (o in r.risk.objects) {
            val d = o.track.det
            stroke.color = RISK_COLORS.getValue(o.level)
            stroke.strokeWidth = (if (o.level == Level.CLEAR) 3f else 6f) * s
            c.drawRect(d.x1, d.y1, d.x2, d.y2, stroke)
            var label = "ID ${o.track.id} ${r.cocoNames[d.cls]} %.2f".format(d.score)
            o.ttc?.let { label += " TTC~%.1fs".format(it) } ?: o.cutInEta?.let { label += " ENTRY~%.1fs".format(it) }
            drawLabel(c, label, d.x1, d.y1, stroke.color, text, s)
        }
        stroke.strokeWidth = 4f * s; stroke.color = SIGN_COLOR
        for ((d, reading) in r.signDets) {
            c.drawRect(d.x1, d.y1, d.x2, d.y2, stroke)
            drawLabel(c, reading.kind.uppercase() + (reading.value?.let { " $it" } ?: ""), d.x1, d.y1, SIGN_COLOR, text, s)
        }

        val bannerH = 96f * s
        fill.color = RISK_COLORS.getValue(r.risk.level); c.drawRect(0f, 0f, w.toFloat(), bannerH * 0.55f, fill)
        fill.color = Color.argb(200, 15, 18, 24); c.drawRect(0f, bannerH * 0.55f, w.toFloat(), bannerH, fill)
        text.textSize = 30f * s
        c.drawText("COLLISION RISK: ${r.risk.level}  |  ${r.risk.reason.take(48)}", 14f * s, 40f * s, text)
        text.textSize = 26f * s; text.color = SIGN_COLOR
        c.drawText("ROAD SIGNS: ${RoadSigns.text(r.confirmedSigns)}", 14f * s, 86f * s, text)

        fill.color = Color.argb(200, 15, 18, 24); c.drawRect(0f, h - 44f * s, w.toFloat(), h.toFloat(), fill)
        text.textSize = 22f * s; text.color = Color.WHITE; text.isFakeBoldText = false
        val people = r.risk.objects.count { it.objectType == "pedestrian" }
        c.drawText(
            "%.1f FPS | %s | %d ms | people %d | objects %d | lane %s | night %s".format(
                fps, r.mode.label, r.inferenceMs, people, r.risk.objects.size, r.lane.status, if (r.night) "ON" else "off"),
            14f * s, h - 14f * s, text)
        return bmp
    }

    private fun drawLabel(c: Canvas, label: String, x: Float, y: Float, color: Int, text: Paint, s: Float) {
        text.textSize = 22f * s; text.color = Color.BLACK
        val tw = text.measureText(label)
        val top = maxOf(0f, y - 30f * s)
        c.drawRect(x, top, x + tw + 12f * s, top + 30f * s, Paint().apply { this.color = color })
        c.drawText(label, x + 6f * s, top + 23f * s, text)
        text.color = Color.WHITE
    }
}
