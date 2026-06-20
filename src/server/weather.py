# src/server/weather.py
# 네이버 날씨 웹 스크래핑 - 환자 외출/방문 안내용

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _extract_digits(tag, label: str = "") -> str:
    """태그 텍스트에서 라벨(예: '최저기온')을 제거하고 숫자만 추출"""
    if not tag:
        return "-"
    text = tag.get_text(strip=True).replace(label, "")
    digits = "".join(c for c in text if c.isdigit() or c in ".-")
    return digits or "-"


def get_naver_weather(location: str = "서울") -> dict:
    """
    네이버 날씨 검색 결과에서 현재 기온/날씨 상태를 스크래핑
    반환: {"temperature": "23", "description": "맑음"} 또는 실패 시 "-" 값
    """
    result = {"temperature": "-", "description": "-"}
    try:
        url = f"https://search.naver.com/search.naver?query={location}+날씨"
        res = requests.get(url, headers=HEADERS, timeout=5)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        temp_tag = soup.find("div", class_="temperature_text")
        if temp_tag:
            text = temp_tag.get_text(strip=True)
            digits = "".join(c for c in text if c.isdigit() or c in ".-")
            result["temperature"] = digits

        desc_tag = soup.find("span", class_="weather before_slash")
        if desc_tag:
            result["description"] = desc_tag.get_text(strip=True)

    except (requests.RequestException, AttributeError, ValueError) as e:
        print(f"[Weather] 스크래핑 실패: {e}")

    return result


def get_naver_weekly_weather(location: str = "서울") -> list:
    """
    주간(향후 7일) 날씨 - 날짜 + 최저/최고 기온만 스크래핑
    반환: [{"date": "6.21(토)", "temp_min": "18", "temp_max": "26"}, ...]
    """
    result = []
    try:
        url = f"https://search.naver.com/search.naver?query={location}+주간날씨"
        res = requests.get(url, headers=HEADERS, timeout=5)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        items = soup.select("li.week_item")[:7]
        for item in items:
            day_tag = item.select_one(".cell_date .day")
            date_tag = item.select_one(".cell_date .date")
            min_tag = item.select_one(".cell_temperature .lowest")
            max_tag = item.select_one(".cell_temperature .highest")
            if not date_tag:
                continue

            day_text = day_tag.get_text(strip=True) if day_tag else ""
            date_text = date_tag.get_text(strip=True) if date_tag else ""

            # 오전/오후 날씨 상태 (구름많음, 비, 눈 등) - 중복되면 하나만 표시
            conditions = []
            for blind in item.select(".cell_weather .wt_icon .blind"):
                text = blind.get_text(strip=True)
                if text and text not in conditions:
                    conditions.append(text)
            condition_text = " / ".join(conditions) if conditions else "-"

            result.append({
                "date": f"{day_text} {date_text}".strip(),
                "temp_min": _extract_digits(min_tag, "최저기온"),
                "temp_max": _extract_digits(max_tag, "최고기온"),
                "condition": condition_text,
            })

    except (requests.RequestException, AttributeError, ValueError) as e:
        print(f"[Weather] 주간 날씨 스크래핑 실패: {e}")

    return result


def get_outing_guide(temperature: str) -> str:
    """
    기온 기준 외출/방문 안내 메시지 생성
    """
    try:
        t = float(temperature)
    except (ValueError, TypeError):
        return ""

    if t >= 33:
        return "폭염 주의 — 외출을 자제하세요"
    elif t <= -10:
        return "한파 주의 — 외출 시 보온에 유의하세요"
    elif t >= 28:
        return "더운 날씨 — 외출 시 수분 섭취에 유의하세요"
    else:
        return "외출하기 적당한 날씨입니다"