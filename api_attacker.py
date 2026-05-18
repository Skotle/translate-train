import requests

key = "https://irisen25.com"

for x in range(1000):
    try:
        # 0.000001초만 기다리고 바로 에러를 발생시켜 다음 코드로 진행
        requests.get(key, timeout=0.2)
    except requests.exceptions.ReadTimeout:
        # 타임아웃 에러가 나더라도 무시하고 진행
        pass
    except requests.exceptions.RequestException as e:
        print(f"연결 오류: {e}")

    print(f"요청 {x + 1}회 전송 완료 (응답 대기 안 함)")