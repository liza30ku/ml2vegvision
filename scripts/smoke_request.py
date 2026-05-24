import httpx

try:
    with open('test_images/Anthracnose (1).jpg','rb') as f:
        r = httpx.post('http://127.0.0.1:8000/predict', files={'image':('Anthracnose.jpg', f, 'image/jpeg')}, timeout=120)
        print('STATUS', r.status_code)
        print(r.text)
except Exception as e:
    print('ERROR', e)
