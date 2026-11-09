package com.old3n.autopilot

/** Port of adas/road_signs.py: sign readings from detector class names only. */
data class SignReading(val kind: String, val value: Int? = null)

object RoadSigns {
    private val STOP_NAMES = setOf("stop sign", "stop")
    private val SPEED_NAME = Regex("^speed_(\\d+)$")
    const val SPEED_MIN_KMH = 10
    const val SPEED_MAX_KMH = 50
    private val ITSD = mapOf(
        "d-xx-1" to SignReading("stop"),
        "d-r-30" to SignReading("speed", 30), "d-r-31" to SignReading("speed", 40),
        "d-r-32" to SignReading("speed", 50), "d-r-33" to SignReading("speed", 60),
        "d-r-34" to SignReading("speed", 70), "d-r-35" to SignReading("speed", 80),
        "d-r-36" to SignReading("speed", 90), "d-r-37" to SignReading("speed", 100),
        "d-r-38" to SignReading("speed", 110), "d-r-39" to SignReading("speed", 120),
        "w-r- intersection" to SignReading("junction"),
        "r-g-k" to SignReading("route"),
    )

    fun kind(name: String): SignReading? {
        val r = ITSD[name]
            ?: if (name in STOP_NAMES) SignReading("stop")
            else SPEED_NAME.find(name)?.let { SignReading("speed", it.groupValues[1].toInt()) }
                ?: if (name.startsWith("junction_")) SignReading("junction") else null
        if (r?.kind == "speed" && r.value!! !in SPEED_MIN_KMH..SPEED_MAX_KMH) return null
        return r
    }

    fun text(confirmed: Set<SignReading>): String {
        val parts = listOf("stop" to "STOP", "junction" to "JUNCTION", "route" to "ROUTE")
            .filter { SignReading(it.first) in confirmed }.map { it.second } +
            confirmed.filter { it.kind == "speed" }.mapNotNull { it.value }.sorted().map { "SPEED $it" }
        return if (parts.isEmpty()) "Signs: none" else parts.joinToString(" | ")
    }
}

/** Accept a reading only when it appears in [frames] consecutive frames. */
class SignConfirmer(private val frames: Int = 3) {
    init { require(frames >= 1) }
    private val history = ArrayDeque<Set<SignReading>>()

    fun reset() = history.clear()

    fun update(readings: Set<SignReading>): Set<SignReading> {
        history.addLast(readings)
        if (history.size > frames) history.removeFirst()
        if (history.size < frames) return emptySet()
        return history.reduce { a, b -> a intersect b }
    }
}
