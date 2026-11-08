import sys, threading, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from AutoPilot import make_server
from playwright.sync_api import sync_playwright
from PIL import Image

class Detector:
    weights='yolo26n.pt'
    device='cpu'
    def predict(self, data, confidence):
        return {'detections':[{'label':'person','confidence':.96,'class_id':0,'bbox':[60,50,200,280]}], 'inference_ms':24.5,'width':640,'height':360,'model':self.weights}

temporary = tempfile.TemporaryDirectory()
output = Path(temporary.name)
server=make_server('127.0.0.1',0,Detector(),'browser-test')
threading.Thread(target=server.serve_forever,daemon=True).start()
url=f'http://127.0.0.1:{server.server_port}/#key=browser-test'
Image.new('RGB',(640,360),'#42514d').save(str(output / 'ui-input.jpg'))
with sync_playwright() as p:
    browser=p.chromium.launch(args=['--no-sandbox','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream'])
    for name, size in [('desktop',{'width':1440,'height':1000}),('mobile',{'width':390,'height':844})]:
        context=browser.new_context(viewport=size,permissions=['camera'])
        page=context.new_page()
        errors=[]
        page.on('pageerror',lambda error: errors.append(str(error)))
        page.goto(url)
        page.get_by_text('Backend connected',exact=True).wait_for()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert page.evaluate('document.querySelector("#empty").scrollWidth <= document.querySelector("#viewport").clientWidth')
        page.screenshot(path=str(output / f'autopilot-{name}.png'),full_page=True)
        page.locator('#file').set_input_files(str(output / 'ui-input.jpg'))
        page.get_by_text('Image analyzed.',exact=False).wait_for()
        assert page.locator('#count').inner_text()=='1'
        assert page.locator('#detections').inner_text()=='person\n96%'
        with page.expect_download() as download:
            page.locator('#export').click()
        assert json.loads(Path(download.value.path()).read_text())['detections'][0]['label']=='person'
        page.locator('#camera').click()
        page.get_by_text('Inference active.',exact=False).wait_for()
        page.locator('#stop').click()
        assert page.evaluate('document.querySelector("video").srcObject === null')
        page.screenshot(path=str(output / f'autopilot-{name}-result.png'),full_page=True)
        page.evaluate('navigator.serviceWorker.ready')
        page.reload()
        page.get_by_text('Backend connected',exact=True).wait_for()
        context.set_offline(True)
        page.reload()
        page.get_by_text('Offline. The app is available',exact=False).wait_for()
        assert not errors, errors
        context.close()
        print(name, 'layout, image inference, export, camera lifecycle, and offline shell passed')
    browser.close()
server.shutdown()
server.server_close()

temporary.cleanup()
