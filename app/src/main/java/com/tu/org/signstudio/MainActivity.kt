package com.tu.org.signstudio

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

import org.json.JSONArray

import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

import kotlin.math.exp

class MainActivity : AppCompatActivity() {

    // =====================================================
    // CONFIGURACIÓN
    // =====================================================

    private val T = 24
    private val FEATURES_PER_FRAME = 126
    private val INPUT_DIM = 3024

    private val CONF_THRESH = 0.90f

    private val REQUIRED_STABLE_PREDICTIONS = 8
    private val ACCEPT_COOLDOWN_MS = 1500L

    // =====================================================
    // INTERFAZ
    // =====================================================

    private lateinit var previewView: PreviewView

    private lateinit var handOverlay: HandOverlayView

    private lateinit var txtMano: TextView
    private lateinit var txtSena: TextView
    private lateinit var txtConfianza: TextView
    private lateinit var txtConversacion: TextView

    private lateinit var btnMicrofono: Button
    private lateinit var btnLeer: Button
    private lateinit var btnLimpiar: Button

    // =====================================================
    // CÁMARA
    // =====================================================

    private lateinit var cameraExecutor: ExecutorService

    // =====================================================
    // MEDIAPIPE
    // =====================================================

    private var handLandmarker: HandLandmarker? = null

    // =====================================================
    // ONNX
    // =====================================================

    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession

    private var labels: List<String> = emptyList()

    // =====================================================
    // BUFFER
    // =====================================================

    private val frameBuffer =
        ArrayDeque<FloatArray>()

    // =====================================================
    // CONVERSACIÓN
    // =====================================================

    private val conversation =
        mutableListOf<String>()

    private var lastPrediction: String? = null

    private var candidatePrediction: String? = null

    private var candidateCount = 0

    private var lastAcceptedTime = 0L

    // =====================================================
    // VOZ
    // =====================================================

    private lateinit var tts: TextToSpeech

    private var speechRecognizer: SpeechRecognizer? = null

    private lateinit var speechIntent: Intent

    private var micEnabled = false

    // =====================================================
    // PERMISO CÁMARA
    // =====================================================

    private val cameraPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->

            if (granted) {
                startCamera()
            }
        }

    // =====================================================
    // PERMISO MICRÓFONO
    // =====================================================

    private val microphonePermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->

            if (granted) {

                iniciarEscucha()

            } else {

                btnMicrofono.text =
                    "🎤 Sin permiso"

                micEnabled = false
            }
        }

    // =====================================================
    // ON CREATE
    // =====================================================

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(savedInstanceState)

        setContentView(
            R.layout.activity_main
        )

        // =================================================
        // CONECTAR INTERFAZ
        // =================================================

        previewView =
            findViewById(
                R.id.previewView
            )

        handOverlay =
            findViewById(
                R.id.handOverlay
            )

        txtMano =
            findViewById(
                R.id.txtMano
            )

        txtSena =
            findViewById(
                R.id.txtSena
            )

        txtConfianza =
            findViewById(
                R.id.txtConfianza
            )

        txtConversacion =
            findViewById(
                R.id.txtConversacion
            )

        btnMicrofono =
            findViewById(
                R.id.btnMicrofono
            )

        btnLeer =
            findViewById(
                R.id.btnLeer
            )

        btnLimpiar =
            findViewById(
                R.id.btnLimpiar
            )

        // =================================================
        // EXECUTOR CÁMARA
        // =================================================

        cameraExecutor =
            Executors
                .newSingleThreadExecutor()

        // =================================================
        // CARGAR IA
        // =================================================

        try {

            loadLabels()

            setupOnnx()

            setupHandLandmarker()

        } catch (e: Exception) {

            e.printStackTrace()

            txtSena.text =
                "Error inicializando IA"
        }

        // =================================================
        // CONFIGURAR VOZ
        // =================================================

        setupTts()

        setupSpeechRecognizer()

        setupButtons()

        // =================================================
        // CÁMARA
        // =================================================

        checkCameraPermission()
    }

    // =====================================================
    // TEXTO A VOZ
    // =====================================================

    private fun setupTts() {

        tts =
            TextToSpeech(this) { status ->

                if (
                    status ==
                    TextToSpeech.SUCCESS
                ) {

                    val result =
                        tts.setLanguage(
                            Locale(
                                "es",
                                "GT"
                            )
                        )

                    if (
                        result ==
                        TextToSpeech.LANG_MISSING_DATA ||
                        result ==
                        TextToSpeech.LANG_NOT_SUPPORTED
                    ) {

                        tts.setLanguage(
                            Locale(
                                "es",
                                "MX"
                            )
                        )
                    }

                    tts.setSpeechRate(
                        0.9f
                    )
                }
            }
    }

    // =====================================================
    // RECONOCIMIENTO DE VOZ
    // =====================================================

    private fun setupSpeechRecognizer() {

        speechRecognizer =
            SpeechRecognizer
                .createSpeechRecognizer(
                    this
                )

        speechIntent =
            Intent(
                RecognizerIntent
                    .ACTION_RECOGNIZE_SPEECH
            ).apply {

                putExtra(
                    RecognizerIntent
                        .EXTRA_LANGUAGE_MODEL,

                    RecognizerIntent
                        .LANGUAGE_MODEL_FREE_FORM
                )

                putExtra(
                    RecognizerIntent
                        .EXTRA_LANGUAGE,

                    "es-GT"
                )

                putExtra(
                    RecognizerIntent
                        .EXTRA_PARTIAL_RESULTS,

                    true
                )
            }

        speechRecognizer
            ?.setRecognitionListener(

                object :
                    RecognitionListener {

                    override fun onReadyForSpeech(
                        params: Bundle?
                    ) {

                        btnMicrofono.text =
                            "🎤 Escuchando..."
                    }

                    override fun onBeginningOfSpeech() {}

                    override fun onRmsChanged(
                        rmsdB: Float
                    ) {}

                    override fun onBufferReceived(
                        buffer: ByteArray?
                    ) {}

                    override fun onEndOfSpeech() {

                        btnMicrofono.text =
                            "🎤 Mic"

                        micEnabled =
                            false
                    }

                    override fun onError(
                        error: Int
                    ) {

                        btnMicrofono.text =
                            "🎤 Mic"

                        micEnabled =
                            false
                    }

                    override fun onResults(
                        results: Bundle?
                    ) {

                        val textos =
                            results
                                ?.getStringArrayList(
                                    SpeechRecognizer
                                        .RESULTS_RECOGNITION
                                )

                        val texto =
                            textos
                                ?.firstOrNull()

                        if (
                            !texto
                                .isNullOrBlank()
                        ) {

                            conversation.add(
                                texto
                            )

                            txtConversacion.text =
                                "Texto: ${
                                    conversation
                                        .joinToString(" ")
                                }"
                        }

                        btnMicrofono.text =
                            "🎤 Mic"

                        micEnabled =
                            false
                    }

                    override fun onPartialResults(
                        partialResults: Bundle?
                    ) {

                        val textos =
                            partialResults
                                ?.getStringArrayList(
                                    SpeechRecognizer
                                        .RESULTS_RECOGNITION
                                )

                        val parcial =
                            textos
                                ?.firstOrNull()

                        if (
                            !parcial
                                .isNullOrBlank()
                        ) {

                            txtSena.text =
                                "Voz: $parcial"
                        }
                    }

                    override fun onEvent(
                        eventType: Int,
                        params: Bundle?
                    ) {}
                }
            )
    }

    // =====================================================
    // INICIAR MICRÓFONO
    // =====================================================

    private fun iniciarEscucha() {

        if (
            !SpeechRecognizer
                .isRecognitionAvailable(
                    this
                )
        ) {

            btnMicrofono.text =
                "Voz no disponible"

            return
        }

        micEnabled =
            true

        speechRecognizer
            ?.startListening(
                speechIntent
            )

        btnMicrofono.text =
            "🎤 Escuchando..."
    }

    // =====================================================
    // BOTONES
    // =====================================================

    private fun setupButtons() {

        // =================================================
        // LEER
        // =================================================

        btnLeer
            .setOnClickListener {

                val texto =
                    conversation
                        .joinToString(
                            " "
                        )

                if (
                    texto
                        .isNotBlank()
                ) {

                    tts.speak(
                        texto,
                        TextToSpeech
                            .QUEUE_FLUSH,
                        null,
                        "SIGNSTUDIO_TTS"
                    )
                }
            }

        // =================================================
        // LIMPIAR
        // =================================================

        btnLimpiar
            .setOnClickListener {

                conversation.clear()

                frameBuffer.clear()

                lastPrediction =
                    null

                candidatePrediction =
                    null

                candidateCount =
                    0

                lastAcceptedTime =
                    0L

                handOverlay
                    .clearLandmarks()

                txtConversacion.text =
                    "Texto:"

                txtSena.text =
                    "Seña: ---"

                txtConfianza.text =
                    "Confianza: ---"
            }

        // =================================================
        // MICRÓFONO
        // =================================================

        btnMicrofono
            .setOnClickListener {

                if (
                    micEnabled
                ) {

                    speechRecognizer
                        ?.cancel()

                    micEnabled =
                        false

                    btnMicrofono.text =
                        "🎤 Mic OFF"

                } else {

                    if (
                        ContextCompat
                            .checkSelfPermission(
                                this,
                                Manifest
                                    .permission
                                    .RECORD_AUDIO
                            ) ==
                        PackageManager
                            .PERMISSION_GRANTED
                    ) {

                        iniciarEscucha()

                    } else {

                        microphonePermissionLauncher
                            .launch(
                                Manifest
                                    .permission
                                    .RECORD_AUDIO
                            )
                    }
                }
            }
    }

    // =====================================================
    // CARGAR LABELS
    // =====================================================

    private fun loadLabels() {

        val jsonText =
            assets
                .open(
                    "labels.json"
                )
                .bufferedReader()
                .use {
                    it.readText()
                }

        val array =
            JSONArray(
                jsonText
            )

        val temp =
            mutableListOf<String>()

        for (
        i in 0 until
                array.length()
        ) {

            var label =
                array
                    .getString(
                        i
                    )

            label =
                label.replace(
                    "\uFEFF",
                    ""
                )

            temp.add(
                label
            )
        }

        labels =
            temp
    }

    // =====================================================
    // ONNX
    // =====================================================

    private fun setupOnnx() {

        ortEnvironment =
            OrtEnvironment
                .getEnvironment()

        val modelBytes =
            assets
                .open(
                    "sign_classifier.onnx"
                )
                .use {
                    it.readBytes()
                }

        ortSession =
            ortEnvironment
                .createSession(
                    modelBytes
                )
    }

    // =====================================================
    // MEDIAPIPE
    // =====================================================

    private fun setupHandLandmarker() {

        val baseOptions =
            BaseOptions
                .builder()
                .setModelAssetPath(
                    "hand_landmarker.task"
                )
                .build()

        val options =
            HandLandmarker
                .HandLandmarkerOptions
                .builder()

                .setBaseOptions(
                    baseOptions
                )

                .setNumHands(
                    2
                )

                .setMinHandDetectionConfidence(
                    0.65f
                )

                .setMinHandPresenceConfidence(
                    0.65f
                )

                .setMinTrackingConfidence(
                    0.65f
                )

                .setRunningMode(
                    RunningMode
                        .LIVE_STREAM
                )

                .setResultListener {
                        result,
                        _ ->

                    processHandResult(
                        result
                    )
                }

                .setErrorListener {
                        error ->

                    error
                        .printStackTrace()

                    runOnUiThread {

                        txtMano.text =
                            "Error MediaPipe"
                    }
                }

                .build()

        handLandmarker =
            HandLandmarker
                .createFromOptions(
                    this,
                    options
                )
    }

    // =====================================================
    // RESULTADOS MEDIAPIPE
    // =====================================================

    private fun processHandResult(
        result: HandLandmarkerResult
    ) {

        val hands =
            result.landmarks()

        // =================================================
        // NO HAY MANOS
        // =================================================

        if (
            hands.isEmpty()
        ) {

            frameBuffer.clear()

            lastPrediction =
                null

            candidatePrediction =
                null

            candidateCount =
                0

            runOnUiThread {

                handOverlay
                    .clearLandmarks()

                txtMano.text =
                    "Mano: No detectada"

                txtSena.text =
                    "Seña: ---"

                txtConfianza.text =
                    "Confianza: ---"
            }

            return
        }

        // =================================================
        // DIBUJAR 21 PUNTOS
        // =================================================

        val firstHand =
            hands[0]

        val overlayPoints =
            firstHand.map {
                    landmark ->

                Pair(
                    landmark.x(),
                    landmark.y()
                )
            }

        runOnUiThread {

            handOverlay
                .setLandmarks(
                    overlayPoints
                )

            txtMano.text =
                "Mano: Detectada (${hands.size})"
        }

        // =================================================
        // EXTRAER FEATURES
        // =================================================

        val features =
            build126Features(
                result
            )

        frameBuffer
            .addLast(
                features
            )

        while (
            frameBuffer.size >
            T
        ) {

            frameBuffer
                .removeFirst()
        }

        // =================================================
        // ESPERAR 24 FRAMES
        // =================================================

        if (
            frameBuffer.size <
            T
        ) {

            runOnUiThread {

                txtSena.text =
                    "Seña: Analizando..."

                txtConfianza.text =
                    "Frames: ${
                        frameBuffer.size
                    }/$T"
            }

            return
        }

        runClassifier()
    }

    // =====================================================
    // CREAR 126 FEATURES
    // =====================================================

    private fun build126Features(
        result: HandLandmarkerResult
    ): FloatArray {

        val left =
            FloatArray(
                63
            )

        val right =
            FloatArray(
                63
            )

        val hands =
            result.landmarks()

        val handedness =
            result.handedness()

        for (
        i in
        hands.indices
        ) {

            val landmarks =
                hands[i]

            if (
                landmarks.size !=
                21
            ) {

                continue
            }

            val normalized =
                normalizeHand(

                    landmarks.map {

                        Triple(
                            it.x(),
                            it.y(),
                            it.z()
                        )
                    }
                )

            var handName =
                ""

            if (
                i <
                handedness.size &&
                handedness[i]
                    .isNotEmpty()
            ) {

                handName =
                    handedness[i][0]
                        .categoryName()
            }

            if (
                handName.equals(
                    "Left",
                    ignoreCase = true
                )
            ) {
                normalized.copyInto(right)
            } else {
                normalized.copyInto(left)

            }
        }

        val output =
            FloatArray(
                126
            )

        left.copyInto(
            output,
            destinationOffset =
                0
        )

        right.copyInto(
            output,
            destinationOffset =
                63
        )

        return output
    }

    // =====================================================
    // NORMALIZAR MANO
    // =====================================================

    private fun normalizeHand(
        points:
        List<
                Triple<
                        Float,
                        Float,
                        Float
                        >
                >
    ): FloatArray {

        val output =
            FloatArray(
                63
            )

        val xs =
            points.map {
                it.first
            }

        val ys =
            points.map {
                it.second
            }

        val xmin =
            xs.minOrNull()
                ?: 0f

        val xmax =
            xs.maxOrNull()
                ?: 1f

        val ymin =
            ys.minOrNull()
                ?: 0f

        val ymax =
            ys.maxOrNull()
                ?: 1f

        val bw =
            maxOf(
                xmax - xmin,
                0.000001f
            )

        val bh =
            maxOf(
                ymax - ymin,
                0.000001f
            )

        var index =
            0

        for (
        point in
        points
        ) {

            val x =
                (
                        point.first -
                                xmin
                        ) / bw

            val y =
                (
                        point.second -
                                ymin
                        ) / bh

            val z =
                point.third

            output[index++] =
                x

            output[index++] =
                y

            output[index++] =
                z
        }

        return output
    }

    // =====================================================
    // CLASIFICADOR
    // =====================================================

    private fun runClassifier() {

        var inputTensor:
                OnnxTensor? =
            null

        var result:
                OrtSession.Result? =
            null

        try {

            val flattened =
                FloatArray(
                    INPUT_DIM
                )

            var offset =
                0

            for (
            frame in
            frameBuffer
            ) {

                frame.copyInto(
                    flattened,
                    destinationOffset =
                        offset
                )

                offset +=
                    FEATURES_PER_FRAME
            }

            // =============================================
            // CREAR TENSOR
            // =============================================

            inputTensor =
                OnnxTensor
                    .createTensor(
                        ortEnvironment,
                        FloatBuffer.wrap(
                            flattened
                        ),
                        longArrayOf(
                            1,
                            INPUT_DIM
                                .toLong()
                        )
                    )

            val inputName =
                ortSession
                    .inputNames
                    .first()

            result =
                ortSession.run(
                    mapOf(
                        inputName
                                to
                                inputTensor
                    )
                )

            // =============================================
            // SALIDA
            // =============================================

            val rawOutput =
                result[0]
                    .value

            val logits =
                when (
                    rawOutput
                ) {

                    is Array<*> -> {

                        @Suppress(
                            "UNCHECKED_CAST"
                        )

                        (
                                rawOutput
                                        as
                                        Array<FloatArray>
                                )[0]
                    }

                    else -> {

                        throw
                        IllegalStateException(
                            "Salida ONNX inesperada"
                        )
                    }
                }

            // =============================================
            // SOFTMAX
            // =============================================

            val probabilities =
                softmax(
                    logits
                )

            var bestIndex =
                0

            var bestConfidence =
                0f

            for (
            i in
            probabilities.indices
            ) {

                if (
                    probabilities[i] >
                    bestConfidence
                ) {

                    bestConfidence =
                        probabilities[i]

                    bestIndex =
                        i
                }
            }

            // =============================================
            // LABEL
            // =============================================

            val label =
                if (
                    bestIndex <
                    labels.size
                ) {

                    labels[
                        bestIndex
                    ]

                } else {

                    "DESCONOCIDO"
                }

            // =============================================
            // CONFIANZA
            // =============================================

            if (
                bestConfidence >=
                CONF_THRESH
            ) {

                runOnUiThread {

                    txtSena.text =
                        "Seña: $label"

                    txtConfianza.text =
                        "Confianza: ${
                            String.format(
                                Locale.US,
                                "%.0f",
                                bestConfidence *
                                        100
                            )
                        }%"
                }

                // =========================================
                // ESTABILIDAD
                // =========================================

                if (
                    candidatePrediction ==
                    label
                ) {

                    candidateCount++

                } else {

                    candidatePrediction =
                        label

                    candidateCount =
                        1
                }

                // =========================================
                // ACEPTAR SEÑA
                // =========================================

                if (
                    candidateCount >=
                    REQUIRED_STABLE_PREDICTIONS
                ) {

                    val currentTime =
                        System
                            .currentTimeMillis()

                    val cooldownPassed =
                        currentTime -
                                lastAcceptedTime >=
                                ACCEPT_COOLDOWN_MS

                    if (
                        lastPrediction !=
                        label &&
                        cooldownPassed
                    ) {

                        lastPrediction =
                            label

                        lastAcceptedTime =
                            currentTime

                        conversation.add(

                            label.replace(
                                "_",
                                " "
                            )
                        )

                        runOnUiThread {

                            txtConversacion.text =
                                "Texto: ${
                                    conversation
                                        .joinToString(
                                            " "
                                        )
                                }"
                        }
                    }
                }

            } else {

                candidatePrediction =
                    null

                candidateCount =
                    0

                runOnUiThread {

                    txtSena.text =
                        "Seña: No reconocida"

                    txtConfianza.text =
                        "Confianza: ${
                            String.format(
                                Locale.US,
                                "%.0f",
                                bestConfidence *
                                        100
                            )
                        }%"
                }
            }

        } catch (
            e: Exception
        ) {

            e.printStackTrace()

            runOnUiThread {

                txtSena.text =
                    "Error ONNX"
            }

        } finally {

            inputTensor
                ?.close()

            result
                ?.close()
        }
    }

    // =====================================================
    // SOFTMAX
    // =====================================================

    private fun softmax(
        logits:
        FloatArray
    ): FloatArray {

        val max =
            logits
                .maxOrNull()
                ?: 0f

        val expValues =
            FloatArray(
                logits.size
            )

        var sum =
            0.0

        for (
        i in
        logits.indices
        ) {

            val value =
                exp(
                    (
                            logits[i] -
                                    max
                            ).toDouble()
                )

            expValues[i] =
                value
                    .toFloat()

            sum +=
                value
        }

        if (
            sum ==
            0.0
        ) {

            return expValues
        }

        for (
        i in
        expValues.indices
        ) {

            expValues[i] =
                (
                        expValues[i] /
                                sum
                        ).toFloat()
        }

        return expValues
    }

    // =====================================================
    // PERMISO CÁMARA
    // =====================================================

    private fun checkCameraPermission() {

        if (
            ContextCompat
                .checkSelfPermission(
                    this,
                    Manifest
                        .permission
                        .CAMERA
                ) ==
            PackageManager
                .PERMISSION_GRANTED
        ) {

            startCamera()

        } else {

            cameraPermissionLauncher
                .launch(
                    Manifest
                        .permission
                        .CAMERA
                )
        }
    }

    // =====================================================
    // INICIAR CÁMARA
    // =====================================================

    private fun startCamera() {

        val cameraProviderFuture =
            ProcessCameraProvider
                .getInstance(
                    this
                )

        cameraProviderFuture
            .addListener({

                val cameraProvider =
                    cameraProviderFuture
                        .get()

                val preview =
                    Preview
                        .Builder()
                        .build()

                preview
                    .setSurfaceProvider(
                        previewView
                            .surfaceProvider
                    )

                val imageAnalysis =
                    ImageAnalysis
                        .Builder()
                        .setBackpressureStrategy(
                            ImageAnalysis
                                .STRATEGY_KEEP_ONLY_LATEST
                        )
                        .build()

                imageAnalysis
                    .setAnalyzer(
                        cameraExecutor
                    ) {
                            imageProxy ->

                        analyzeImage(
                            imageProxy
                        )
                    }

                // Cámara frontal

                val cameraSelector =
                    CameraSelector
                        .DEFAULT_FRONT_CAMERA

                try {

                    cameraProvider
                        .unbindAll()

                    cameraProvider
                        .bindToLifecycle(
                            this,
                            cameraSelector,
                            preview,
                            imageAnalysis
                        )

                } catch (
                    e: Exception
                ) {

                    e.printStackTrace()
                }

            },
                ContextCompat
                    .getMainExecutor(
                        this
                    )
            )
    }

    // =====================================================
    // ANALIZAR IMAGEN
    // =====================================================

    private fun analyzeImage(
        imageProxy:
        ImageProxy
    ) {

        try {

            var bitmap =
                imageProxy
                    .toBitmap()

            val rotationDegrees =
                imageProxy
                    .imageInfo
                    .rotationDegrees

            // =============================================
            // ROTACIÓN
            // =============================================

            if (
                rotationDegrees !=
                0
            ) {

                val rotationMatrix =
                    Matrix()

                rotationMatrix
                    .postRotate(
                        rotationDegrees
                            .toFloat()
                    )

                bitmap =
                    Bitmap
                        .createBitmap(
                            bitmap,
                            0,
                            0,
                            bitmap.width,
                            bitmap.height,
                            rotationMatrix,
                            true
                        )
            }

            // =============================================
            // ESPEJO CÁMARA FRONTAL
            // =============================================

            val mirrorMatrix =
                Matrix()

            mirrorMatrix
                .preScale(
                    -1f,
                    1f
                )

            bitmap =
                Bitmap
                    .createBitmap(
                        bitmap,
                        0,
                        0,
                        bitmap.width,
                        bitmap.height,
                        mirrorMatrix,
                        true
                    )

            // =============================================
            // MEDIAPIPE
            // =============================================

            val mpImage =
                BitmapImageBuilder(
                    bitmap
                ).build()

            val timestamp =
                SystemClock
                    .uptimeMillis()

            handLandmarker
                ?.detectAsync(
                    mpImage,
                    timestamp
                )

        } catch (
            e: Exception
        ) {

            e.printStackTrace()

        } finally {

            imageProxy
                .close()
        }
    }

    // =====================================================
    // CERRAR RECURSOS
    // =====================================================

    override fun onDestroy() {

        super.onDestroy()

        speechRecognizer
            ?.cancel()

        speechRecognizer
            ?.destroy()

        speechRecognizer =
            null

        if (
            ::tts.isInitialized
        ) {

            tts.stop()

            tts.shutdown()
        }

        handLandmarker
            ?.close()

        handLandmarker =
            null

        if (
            ::ortSession
                .isInitialized
        ) {

            ortSession
                .close()
        }

        if (
            ::cameraExecutor
                .isInitialized
        ) {

            cameraExecutor
                .shutdown()
        }
    }
}