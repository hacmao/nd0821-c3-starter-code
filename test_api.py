import requests

API_URL = "https://python-ml-api.herokuapp.com/"


def test_get():
    response = requests.get(API_URL)
    assert response.status_code == 200
    assert response.json()["message"] == "welcome"


def test_post_return_0():
    response = requests.post(
        API_URL,
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        },
        timeout=15
    )
    assert response.status_code == 200
    assert response.json()['predict'] == 0


def test_post_return_1():
    response = requests.post(
        API_URL,
        json={
            "age": 31,
            "workclass": "Private",
            "fnlgt": 45781,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 14084,
            "capital-loss": 0,
            "hours-per-week": 50,
            "native-country": "string"
        },
        timeout=15
    )
    assert response.status_code == 200
    assert response.json()['predict'] == 1


if __name__ == "__main__":
    test_get()
    test_post_return_0()
    test_post_return_1()
