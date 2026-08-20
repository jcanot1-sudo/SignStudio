package com.tu.org.signstudio

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class HandOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var points: List<Pair<Float, Float>> = emptyList()

    // Pincel para los puntos
    private val pointPaint = Paint().apply {
        style = Paint.Style.FILL
        strokeWidth = 8f
        isAntiAlias = true
    }

    // Pincel para las líneas
    private val linePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        isAntiAlias = true
    }

    // Conexiones de los 21 puntos de MediaPipe
    private val connections = listOf(

        // Pulgar
        0 to 1,
        1 to 2,
        2 to 3,
        3 to 4,

        // Índice
        0 to 5,
        5 to 6,
        6 to 7,
        7 to 8,

        // Medio
        5 to 9,
        9 to 10,
        10 to 11,
        11 to 12,

        // Anular
        9 to 13,
        13 to 14,
        14 to 15,
        15 to 16,

        // Meñique
        13 to 17,
        17 to 18,
        18 to 19,
        19 to 20,

        // Palma
        0 to 17
    )

    // =====================================================
    // RECIBIR LOS PUNTOS DE MEDIAPIPE
    // =====================================================

    fun setLandmarks(
        newPoints: List<Pair<Float, Float>>
    ) {

        points = newPoints

        invalidate()
    }

    // =====================================================
    // LIMPIAR LA MANO
    // =====================================================

    fun clearLandmarks() {

        points = emptyList()

        invalidate()
    }

    // =====================================================
    // DIBUJAR
    // =====================================================

    override fun onDraw(
        canvas: Canvas
    ) {

        super.onDraw(canvas)

        // MediaPipe debe entregar 21 puntos
        if (points.size != 21) {
            return
        }

        // =============================================
        // DIBUJAR LÍNEAS
        // =============================================

        for (
        connection in connections
        ) {

            val start =
                connection.first

            val end =
                connection.second

            val p1 =
                points[start]

            val p2 =
                points[end]

            val x1 =
                p1.first * width

            val y1 =
                p1.second * height

            val x2 =
                p2.first * width

            val y2 =
                p2.second * height

            canvas.drawLine(
                x1,
                y1,
                x2,
                y2,
                linePaint
            )
        }

        // =============================================
        // DIBUJAR LOS 21 PUNTOS
        // =============================================

        for (
        point in points
        ) {

            val x =
                point.first * width

            val y =
                point.second * height

            canvas.drawCircle(
                x,
                y,
                10f,
                pointPaint
            )
        }
    }
}