# code

## UVCCamera

Library and sample to access UVC web camera on non-rooted Android devices.

**Source:** https://github.com/saki4510t/UVCCamera

### Overview

UVCCamera is a library that enables Android applications to access USB Video Class (UVC) web cameras on non-rooted Android devices. It provides a native implementation for communicating with UVC-compliant cameras via USB host mode.

### Requirements

- Android 3.1 or later (API >= 12), Android 4.0+ (API >= 14) recommended
- USB host function support on the device
- Some sample projects require API >= 18

### How to Compile

#### Using Gradle

1. Create a directory for the project
2. Clone the repository: `git clone https://github.com/saki4510t/UVCCamera.git`
3. Navigate to the `UVCCamera` directory: `cd UVCCamera`
4. Build the library: `./gradlew build`
5. APKs will be available in `{sample project}/build/outputs/apks`
6. To install all samples: `./gradlew installDebug`

**Note:** Ensure `local.properties` contains paths for `sdk.dir` and `ndk.dir`, or set them as environment variables. You may also need to set `JAVA_HOME`.

#### Using Android Studio

1. Clone the repository
2. Open the project in Android Studio using "Open an existing Android Studio project"
3. Add `ndk.dir` to `local.properties`:
   ```
   sdk.dir={path to Android SDK}
   ndk.dir={path to Android NDK}
   ```
4. Synchronize and build the project

#### Using NDK Build (Eclipse/Legacy)

1. Clone the repository
2. Navigate to `{UVCCamera}/libuvccamera/build/src/main/jni`
3. Run `ndk-build`
4. Libraries will be in `{UVCCamera}/libuvccamera/build/src/main/libs`

### Sample Projects

- **USBCameraTest2** - Movie capture using MediaCodec and MediaMuxer (API >= 18)
- **USBCameraTest3** - Audio and video capture with still image support (API >= 18)
- **USBCameraTest4** - Offscreen rendering and background service recording
- **USBCameraTest5** - IFrameCallback interface for frame data as ByteArray
- **USBCameraTest6** - Dual TextureView display (side by side)
- **USBCameraTest7** - Dual camera support
- **USBCameraTest8** - UVC control settings (brightness, contrast, etc.)

### Features

- Access UVC cameras without root
- Video preview and recording
- Still image capture (PNG/JPEG)
- Camera control (brightness, contrast, etc.)
- Dual camera support
- Stereo camera support (experimental)
- Android N (7.x) support with dynamic permissions

### License

Apache License, Version 2.0

Copyright (c) 2014-2017 saki t_saki@serenegiant.com

Note: Files in `jni/libjpeg`, `jni/libusb`, and `jni/libuvc` folders may have different licenses.
