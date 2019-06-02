import torch

from flask import Flask, request, jsonify
from model.encoder_decoder import EncoderDecoder
import json
import logging

log_server = logging.getLogger(__name__)
log_server.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='test.log', encoding='utf-8')
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)8s %(message)s'))
log_server.addHandler(handler)

app = Flask(__name__)
encoder_decoder_for_inter: EncoderDecoder = torch.load('./model/test-model/' + 'test-model' + '_199' + '.pt')


# SERVER
@app.route('/transform', methods=['GET'])
def transformer_server():
    data = json.loads(request.data)
    sentence = data['input']
    pred_line = encoder_decoder_for_inter.get_response(sentence, remove=True)
    data['user-id'] = data.get('user-id', "None")
    data['date'] = data.get('date', "None")
    res = {
        "Content-Type": "application/json",
        "user-id": data['user-id'],
        "request-date": data['date'],
        "input": data['input'],
        "turn": pred_line,
    }
    log_server.info('{}'.format(res))
    return jsonify(res)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10204, debug=False)

