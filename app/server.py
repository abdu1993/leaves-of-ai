from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import sys
from pathlib import Path
import csv
import time

from fastai import *
from fastai.text import *

model_file_url = 'https://drive.google.com/uc?export=download&id=1iM4LYs9wmBnZi9auIxF0iWmiGTpbc63-'
model_file_name = 'model_2_10'
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_lm = TextLMDataBunch.load(path/'static', 'data_lm')
    data_bunch = (TextList.from_csv(path, csv_name='static/blank.csv', vocab=data_lm.vocab)
        .random_split_by_pct()
        .label_for_lm()
        .databunch(bs=10))
    learn = language_model_learner(data_bunch, pretrained_model=None)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()

    return JSONResponse({'result': textResponse(data)})

def textResponse(data, temp_init=1, decay=50, stops = ['.'], comma_limit=2, words=50):
    seeds = ['you are', 'i love you', 'we are', 'how can']
    word = random.choice(seeds)    
    again = []
    stops.append('xxbos')
    last_4 = ['','','','']
    total = []
    stops.append('xxbos')
    i = 0
    commas = 0
    temp = temp_init
    while True:
        if any(x in word for x in stops):
           break
        if commas > comma_limit:
            break    
        if all(x in (" ".join([i for i in word.split()[-9:-5]])) for x in last_4) and (len(word.split()) > 10):
            break
        addition = learn.predict(word, 1, temperature=temp)
        total.append(addition)
        commas = addition.count(',')
        last_4 = total[i].split()[-4:]
        word = addition
        i += 1
        temp = max(abs((math.cos(len(total) * math.pi / decay)))*temp_init,0.1)

    words = total[-2].split()
    for i, word in enumerate(words):
        if word == 'xxbos':
            words[i] = '<br/>'
        elif word == 'xxmaj':
            words[i+1] = words[i+1][0].upper() + words[i+1][1:]
            words[i] = ''
        elif word == 'xxup':
            words[i+1] = words[i+1].upper()
            words[i] = ''     
        elif word == 'xxunk' or word == '(' or word == ')' or word == '"':
            words[i] = ''   
        elif word == "n't":
            words[i-1]+= words[i]
            words[i] = ''
        elif word == ",":
            words[i-1]+= words[i]
            words[i] = ''
        elif word == '.' or word == '?' or word == '!' or word == ';':
            words[i-1]+= words[i]
            words[i] = ''
        elif word[0] == "'":
            words[i-1]+= words[i]
            words[i] = ''
        elif word == ' ':
            words[i] = ''
        elif "n't" in word:
            words[i-1]+= words[i]
            words[i] = ''
        elif word == "na":
            words[i-1]+= words[i]
            words[i] = ''
        try:
            words.remove('')
        except:
            continue

    return(' '.join(words))

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)

