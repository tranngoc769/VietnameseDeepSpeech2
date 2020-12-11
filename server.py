import logging
import os
from tempfile import NamedTemporaryFile

import hydra
import torch
from flask import Flask, request, jsonify
from hydra.core.config_store import ConfigStore
import time
from deepspeech_pytorch.configs.inference_config import ServerConfig
from deepspeech_pytorch.inference import run_transcribe
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.utils import load_model, load_decoder
from flask import send_file, send_from_directory
from flask import render_template
import glob
app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['.wav', '.mp3', '.ogg', '.webm'])

cs = ConfigStore.instance()
cs.store(name="config", node=ServerConfig)

import time
import datetime
import json
@app.errorhandler(404)
def page_not_found(e):
    request.path = request.path.replace("//", "/")
    if ("/download/" in request.path):
        req = request.path.split("/download/")
        status = True
        try:
            if (req[0] != '' and len(req < 2)):
                status = False
        except:
                status = False
        if status == False:
            return {'code': 403, 'msg': "Invalid"}
        filename = req[len(req)-1]
        path  = "/".join(req)
        path = path.replace("//", "/")
        return send_file(path, attachment_filename = filename, as_attachment=True)
    else:
        req = request.path.split("/file/")
        status = True
        try:
            if (req[0] != '' and len(req < 2)):
                status = False
        except:
                status = False
        
        if status == False:
            return {'code': 403, 'msg': "Invalid"}
        path  = "/".join(req)
        path = path.replace("//", "/")
        res = {}
        res['code'] = 200
        res['message'] = "index"
        files_path = glob.glob(path+"/*")
        if (path==None or len(files_path) < 1):
                res['code'] = 404
                res['message'] = 'No file / folder found'
                return res
        data = []
        for child in os.scandir(path):
            item = {}
            item['path'] = child.path
            item['name'] = child.name
            item['size'] = str( "%.1f" % (child.stat().st_size/ 1024))
            item['date_access'] = datetime.datetime.fromtimestamp(child.stat().st_atime).strftime('%Y/%m/%d %H:%M:%S')
            item['date_create'] = datetime.datetime.fromtimestamp(child.stat().st_mtime).strftime('%Y/%m/%d %H:%M:%S')
            if (os.path.isfile(child.path)):
                item['type'] = 'File'
            else:
                item['type'] = 'Folder'
            data.append(item)
        res['message'] = data
        return render_template('file.html', data = res)
@app.route('/transcribe', methods=['POST'])
def transcribe_file():
    if request.method == 'POST':
        res = {}
        res['total'] = 0
        res['seconds'] = 0
        t0 = time.time()

        if 'file' not in request.files:
            res['code'] = 403
            res['data'] = "Missed audio files"
            return jsonify(res)
        try:
            file = request.files['file']
            filename = file.filename
            _, file_extension = os.path.splitext(filename)
            if file_extension.lower() not in ALLOWED_EXTENSIONS:
                res['code'] = 403
                res['data'] = "{} is not supported format.".format(file_extension)
                return jsonify(res)
            with NamedTemporaryFile(suffix=file_extension) as tmp_saved_audio_file:
                file.save(tmp_saved_audio_file.name)
                logging.info('Transcribing file...')
                transcription, _ = run_transcribe(audio_path=tmp_saved_audio_file,
                                                spect_parser=spect_parser,
                                                model=model,
                                                decoder=decoder,
                                                device=device,
                                                use_half=True)
                logging.info('File transcribed')
                res['status'] = 200
                if (len(transcription) > 0):
                    res['data'] = transcription[0]
                    res['total'] = len(transcription[0])
                else:
                    res['data'] = transcription
                    res['total'] = len(transcription)
        except Exception as exx:
            res['status'] = 200
            res['data'] = str(exx)
        t1 = time.time()
        total = t1-t0
        res['seconds'] = total
        return res
@app.route('/file')
def index(name):
    res = {}
    res['code'] = 200
    res['message'] = "index"
    files_path = glob.glob(name+"/*")
    if (name==None or len(files_path) < 1):
            res['code'] = 404
            res['message'] = 'No file / folder found'
            return res
    data = []
    for child in os.scandir("/work"):
        item = {}
        if (os.path.isfile(child.path)):
            item['path'] = child.path
            item['name'] = child.name
            item['size'] = str( "%.1f" % (child.stat().st_size/ 1024))
            item['date_access'] = datetime.datetime.fromtimestamp(item.stat().st_atime).strftime('%Y/%m/%d %H:%M:%S')
            item['date_create'] = datetime.datetime.fromtimestamp(item.stat().st_mtime).strftime('%Y/%m/%d %H:%M:%S')
            item['date_create'] = datetime.datetime.fromtimestamp(item.stat().st_mtime).strftime('%Y/%m/%d %H:%M:%S')
            item['type'] = 'File'
        else:
            item['path'] = child.path
            item['type'] = 'Folder'
        data.append(item)
    res['message'] = json.loads(data)
    return render_template('index.html')
if __name__ == '__main__':
   app.run(debug = True)
@hydra.main(config_name="config")
def main(cfg: ServerConfig):
    global model, spect_parser, decoder, config, device
    config = cfg
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('Setting up server...')
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(device=device,
                       model_path=cfg.model.model_path,
                       use_half=cfg.model.use_half)

    decoder = load_decoder(labels=model.labels,
                           cfg=cfg.lm)

    spect_parser = SpectrogramParser(audio_conf=model.audio_conf,
                                     normalize=True)

    spect_parser = SpectrogramParser(model.audio_conf, normalize=True)
    logging.info('Server initialised')
    app.run(host=cfg.host, port=cfg.port, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
