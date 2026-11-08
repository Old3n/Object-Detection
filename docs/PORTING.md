# Porting and release instructions

[Quick start](../README.md) · [Research workflows](RESEARCH_GUIDE.md)

## Architecture

`AutoPilot.py` serves `web/` and an authenticated `/api/detect` endpoint on one origin. The browser captures a camera frame or decodes a local image/video and sends JPEG bytes. The Python backend loads the selected Ultralytics checkpoint lazily, returns detection JSON, and keeps a single model behind a nonblocking lock. Concurrent requests receive HTTP 429 rather than mixing model state.

The UI renders the corresponding submitted frame and exports its result. It never caches inference responses. `sw.js` caches only the public application shell, which opens offline but cannot infer without the backend. Camera streams are released on Stop, page exit, and backgrounding. Browser permissions and available codecs vary by device.

The executable contains the dashboard/backend, CPU dependencies, and default checkpoint. Research/training CLI tools remain in the source repository and are not embedded as executable subcommands.

## Windows single-file build

Build on Windows x64 with Python 3.11; PyInstaller does not cross-compile Windows executables from Linux. Use the commands in the README or the **Build desktop and PWA** GitHub Actions workflow.

`AutoPilot.spec` collects Ultralytics package data and hidden imports, the `web/` directory, and `yolo26n.pt`. PyInstaller's runtime root is used to locate bundled resources, independently of the working directory. Use `--weights` to supply an external checkpoint without rebuilding.

A one-file executable still extracts its runtime into a temporary directory on launch. The bundled PyTorch runtime is large and startup is slower than a normal source run. The build uses a console so errors, the access key, and shutdown remain accessible. The binary is unsigned; signing requires a certificate and a separate signing step owned by the distributor.

Run `AutoPilot.exe --self-test` before publishing. This checks bundled assets and performs actual model inference on a synthetic image. Also launch from a different working directory on a clean Windows machine, open an image, try a camera, export JSON, and check shutdown. CI's self-test cannot validate physical cameras or SmartScreen behavior.

## Android PWA deployment

The phone runs the browser UI, while your desktop/server runs Python inference. There is no APK and no on-device inference in this port.

1. Install the backend dependencies and keep the server running:

   ```bash
   python AutoPilot.py --no-browser --host 127.0.0.1 --port 8765
   ```

2. Use a domain you control with a trusted TLS certificate. A reverse proxy on the same computer should forward the entire origin to `127.0.0.1:8765`. For example, with Caddy installed and DNS/certificate prerequisites configured:

   ```text
   detection.example.com {
       reverse_proxy 127.0.0.1:8765
   }
   ```

   Replace the example domain. Public certificate issuance typically needs correct DNS and reachable challenge ports, or a configured DNS challenge. For private networks, use a private VPN/proxy HTTPS endpoint trusted by the phone, or a locally issued certificate whose CA is installed and trusted there. An untrusted self-signed certificate is not sufficient for a normal install flow.

3. Open `https://YOUR_DOMAIN/` on the phone. Enter the access key shown by the backend. The key changes on each restart and is stored in browser session storage, not in the service-worker cache. Re-enter it when the session ends.
4. Use the install button if Chrome exposes it, or Chrome's install menu. Open the installed app, grant camera permission, and select Camera.
5. Verify Stop releases the camera, portrait/landscape layouts fit, image and video selection work, and JSON export downloads successfully.
6. Disconnect the backend/network: the cached interface should open, show that it is disconnected, and avoid displaying new inference results. Reconnect and enter the current key if required.

Keep the backend bound to loopback behind the proxy. A bearer access key is required for API requests, but the stdlib server is intended for personal/trusted use, not a public multi-tenant service. For shared deployment, add a production server, per-user authentication, request limits, and resource isolation. The backend intentionally provides no cross-origin API access; frontend and API must remain on the same origin at `/`. GitHub Pages alone cannot run the Python inference service.

The `AutoPilot-pwa.zip` archive includes `AutoPilot.py`, `requirements.txt`, `web/`, and these instructions. Extract, install requirements, and run the same command. Internet is required to fetch the default checkpoint on first launch, unless you provide an existing checkpoint with `--weights`.

References: [camera secure-context requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia), [service workers](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API), [PyInstaller builds](https://pyinstaller.org/en/stable/).

## Future native desktop ports

1. Build on the target OS and architecture using a clean supported Python environment.
2. Install a PyTorch/OpenCV combination available for that target and run the full tests.
3. Download the chosen checkpoint and run PyInstaller with `AutoPilot.spec`. The result is a platform-native executable (`AutoPilot` on Linux/macOS), not a Windows `.exe`.
4. Run `--self-test` and manually verify browser launch, webcam permission, file selection, and shutdown on that OS.
5. Sign/notarize where required, record `pip freeze`, and publish artifacts named with their OS and architecture. Add a separate CI job for each supported target.

## Future on-device Android inference

A fully offline Android port requires additional model/runtime work:

1. Export the chosen checkpoint to a browser-compatible runtime such as ONNX and confirm supported operators and output layout for that exact model version.
2. Add and bundle a mobile-compatible inference runtime, preprocessing, and postprocessing. Implement the `/api/detect` response contract locally or replace the fetch layer in `web/app.js`.
3. Cache model/runtime assets deliberately and provide download progress, storage checks, and update/version handling.
4. Compare browser predictions against the Python model on fixed labelled fixtures. Measure accuracy, memory, latency, battery impact, and sustained thermal performance on target phones.
5. Test camera lifecycle and offline operation on physical Android devices before claiming device-only support. An APK wrapper/TWA alone does not move Python inference onto the phone.

## Release maintenance

- Run unit tests, JavaScript syntax checks, browser acceptance checks, and real inference before tagging.
- Increment the service-worker cache version when changing the shell's asset list or making incompatible changes.
- Build from an immutable reviewed commit. Use an unused `v*` tag; never repoint an existing release tag.
- Tag builds and `release/v*` branches publish only after both the Windows self-test and PWA packaging finish. Feature-branch/manual builds remain workflow artifacts. A `release/v1.0.0` branch creates the matching `v1.0.0` tag and release at that exact commit; use this once per new version.
- Release assets include `AutoPilot.exe`, `AutoPilot-pwa.zip`, `build-dependencies.txt`, and `SHA256SUMS.txt`. Archive the resolved dependency inventory when reproducing a release.
- Confirm upstream model/dependency licensing and any redistribution requirements. Custom model files must come from a trusted source.
