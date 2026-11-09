plugins {
    id("com.android.application")
}

android {
    namespace = "com.old3n.autopilot"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.old3n.autopilot"
        minSdk = 21  // Android 5.0: the floor for CameraX, ONNX Runtime and OpenCV
        targetSdk = 35
        versionCode = 2
        versionName = "0.2.0"
    }

    // One APK per CPU type keeps each install small. armeabi-v7a covers most
    // 32-bit car head units; x86_64 is only for the emulator.
    splits {
        abi {
            isEnable = true
            reset()
            include("armeabi-v7a", "arm64-v8a", "x86_64")
            isUniversalApk = false
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Signed with the debug key so the APK installs directly for field tests.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources { noCompress += "onnx" }
    packaging { jniLibs { useLegacyPackaging = true } }
}

dependencies {
    val camerax = "1.4.1"
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation("org.opencv:opencv:4.10.0")
    testImplementation("junit:junit:4.13.2")
}
