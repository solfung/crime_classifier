import requests

def test_alive():
    url = 'http://localhost:5000/alive'
    data = {'rate':5, 'sales_in_first_month':200, 'sales_in_second_month':400}
    r = requests.post(url,json=data)
    assert r.json() == data

