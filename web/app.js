const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('output');
const context = canvas.getContext('2d');
const frame = document.createElement('canvas');
const frameContext = frame.getContext('2d');
let key = new URLSearchParams(location.hash.slice(1)).get('key') || sessionStorage.getItem('autopilot-key') || '';
if (location.hash) history.replaceState(null, '', location.pathname);
let generation = 0;
let stream = null;
let fileUrl = null;
let report = null;
let installPrompt = null;
let controller = null;
let connected = false;

function message(text, error = false) {
  $('message').textContent = text;
  $('message').classList.toggle('error', error);
}

async function connect() {
  try {
    const response = await fetch('/api/health', {headers: {Authorization: `Bearer ${key}`}, signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error('Enter the access key printed in the AutoPilot console.');
    const data = await response.json();
    sessionStorage.setItem('autopilot-key', key);
    $('auth').hidden = true;
    $('model').textContent = data.model;
    $('device').textContent = `${data.device.toUpperCase()} · connected inference`;
    $('connection').textContent = 'Backend connected';
    $('connection').classList.add('online');
    connected = true;
    message('Ready. Choose a camera, image, or video to start.');
  } catch (error) {
    connected = false;
    $('auth').hidden = false;
    $('connection').textContent = navigator.onLine ? 'Not connected' : 'Offline';
    $('connection').classList.remove('online');
    message(navigator.onLine ? error.message : 'Offline. The app is available, but detection needs your backend connection.', true);
  }
}

function stop() {
  generation += 1;
  controller?.abort();
  controller = null;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.pause();
  video.srcObject = null;
  video.removeAttribute('src');
  video.load();
  if (fileUrl) URL.revokeObjectURL(fileUrl);
  fileUrl = null;
  $('stop').disabled = true;
  $('feed-label').textContent = 'PAUSED FRAME';
}

function resetResults() {
  report = null;
  $('export').disabled = true;
  $('count').textContent = '—';
  $('latency').textContent = '—';
  $('result-count').textContent = '0';
  $('detections').replaceChildren();
  canvas.hidden = true;
  $('empty').hidden = false;
  $('feed-label').hidden = true;
}

async function detect(source, current) {
  const width = source.videoWidth || source.naturalWidth;
  const height = source.videoHeight || source.naturalHeight;
  if (!width || !height || current !== generation) return;
  const scale = Math.min(1, 1280 / Math.max(width, height));
  frame.width = Math.round(width * scale);
  frame.height = Math.round(height * scale);
  frameContext.drawImage(source, 0, 0, frame.width, frame.height);
  const blob = await new Promise(resolve => frame.toBlob(resolve, 'image/jpeg', 0.85));
  if (current !== generation) return;
  controller = new AbortController();
  const timeout = setTimeout(() => controller?.abort(), 120000);
  let response;
  try {
    response = await fetch(`/api/detect?confidence=${Number($('confidence').value) / 100}`, {
      method: 'POST', headers: {'Content-Type': 'image/jpeg', Authorization: `Bearer ${key}`},
      body: blob, signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
  const data = await response.json();
  if (current !== generation) return;
  if (!response.ok) throw new Error(data.error || 'Detection failed.');
  report = {...data, captured_at: new Date().toISOString(), confidence_threshold: Number($('confidence').value) / 100};
  canvas.width = frame.width;
  canvas.height = frame.height;
  context.drawImage(frame, 0, 0);
  context.lineWidth = Math.max(2, frame.width / 500);
  context.font = `${Math.max(13, frame.width / 65)}px system-ui`;
  data.detections.forEach(item => {
    const [x1, y1, x2, y2] = item.bbox;
    context.strokeStyle = '#d9fa83';
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${item.label} ${Math.round(item.confidence * 100)}%`;
    const textWidth = context.measureText(label).width;
    const top = Math.max(0, y1 - 25);
    context.fillStyle = '#d9fa83';
    context.fillRect(x1, top, textWidth + 12, 25);
    context.fillStyle = '#121717';
    context.fillText(label, x1 + 6, top + 18);
  });
  canvas.hidden = false;
  $('empty').hidden = true;
  $('feed-label').hidden = false;
  $('feed-label').textContent = 'DETECTION VIEW';
  $('dimensions').textContent = `${frame.width} × ${frame.height}`;
  $('count').textContent = data.detections.length;
  $('latency').textContent = `${data.inference_ms} ms`;
  $('result-count').textContent = data.detections.length;
  $('detections').replaceChildren();
  data.detections.slice().sort((a,b) => b.confidence - a.confidence).forEach(item => {
    const row = document.createElement('li');
    const label = document.createElement('span');
    const confidence = document.createElement('span');
    label.textContent = item.label;
    confidence.textContent = `${Math.round(item.confidence * 100)}%`;
    row.append(label, confidence);
    $('detections').append(row);
  });
  if (!data.detections.length) {
    const row = document.createElement('li');
    row.className = 'no-results';
    row.textContent = 'No objects above this threshold.';
    $('detections').append(row);
  }
  $('export').disabled = false;
  message('Inference active. Each result corresponds to the displayed frame.');
}

async function run(current) {
  try {
    while (current === generation && !video.ended) {
      if (video.readyState >= 2) await detect(video, current);
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (current === generation) {
      stop();
      message('Video complete. Export the final frame results or open another source.');
    }
  } catch (error) {
    if (current !== generation) return;
    stop();
    message(error.name === 'AbortError' ? 'Detection timed out. Check the backend and try again.' : error.message, true);
  }
}

async function startCamera() {
  if (!connected) { $('auth').hidden = false; message('Connect to your backend first.', true); return; }
  stop();
  resetResults();
  const current = generation;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS on Android, or localhost on desktop.');
    const requestedStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'environment', width: {ideal: 1280}}, audio: false});
    if (current !== generation) { requestedStream.getTracks().forEach(track => track.stop()); return; }
    stream = requestedStream;
    video.srcObject = stream;
    await video.play();
    if (current !== generation) return;
    $('source-label').textContent = 'INPUT / LIVE CAMERA';
    $('stop').disabled = false;
    message('Loading detector. The first frame can take a moment…');
    run(current);
  } catch (error) {
    if (current === generation) { stop(); message(error.message, true); }
  }
}

$('camera').onclick = $('empty-camera').onclick = startCamera;
$('stop').onclick = () => { stop(); message('Stopped. The camera is released; the last result remains visible.'); };
$('connect').onclick = () => { key = $('key').value.trim(); connect(); };
$('confidence').oninput = () => { $('confidence-value').textContent = `${$('confidence').value}%`; };
$('file').onchange = async (event) => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  if (!connected) { message('Connect to your backend first.', true); return; }
  stop();
  resetResults();
  const current = generation;
  fileUrl = URL.createObjectURL(file);
  $('source-label').textContent = `INPUT / ${file.name}`;
  $('stop').disabled = false;
  message('Loading detector. The first frame can take a moment…');
  try {
    if (file.type.startsWith('video/')) {
      video.src = fileUrl;
      await video.play();
      if (current === generation) run(current);
    } else if (file.type.startsWith('image/')) {
      const image = new Image();
      image.src = fileUrl;
      await image.decode();
      await detect(image, current);
      if (current === generation) { stop(); message('Image analyzed. Export results or choose another input.'); }
    } else throw new Error('Choose an image or a browser-supported video.');
  } catch (error) {
    if (current === generation) { stop(); message(error.message, true); }
  }
};
$('export').onclick = () => {
  if (!report) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'autopilot-frame.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
window.addEventListener('beforeinstallprompt', event => { event.preventDefault(); installPrompt = event; $('install').hidden = false; });
$('install').onclick = async () => { if (installPrompt) { await installPrompt.prompt(); installPrompt = null; $('install').hidden = true; } };
window.addEventListener('appinstalled', () => { $('install').hidden = true; });
window.addEventListener('offline', () => { stop(); connect(); });
window.addEventListener('online', connect);
window.addEventListener('pagehide', stop);
document.addEventListener('visibilitychange', () => { if (document.hidden && !$('stop').disabled) { stop(); message('Paused while the app is in the background.'); } });
if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => message('Offline app caching is unavailable in this browser.'));
connect();
