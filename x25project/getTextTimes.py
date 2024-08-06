from datetime import datetime, timedelta
def getTextTimes(start_date_str, end_date_str):
    # Định dạng ngày tháng năm
    date_format = '%d-%m-%Y'
    textTimes = []
    # Chuyển đổi chuỗi ngày tháng năm thành đối tượng datetime
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    
    # Kiểm tra nếu ngày kết thúc trước ngày bắt đầu
    if end_date < start_date:
        print("Ngày kết thúc không thể trước ngày bắt đầu.")
        return textTimes
    
    # Duyệt qua các ngày từ ngày bắt đầu đến ngày kết thúc
    current_date = start_date
    while current_date <= end_date:
        # Chuyển đổi ngày hiện tại thành chuỗi theo định dạng
        textTime = current_date.strftime(date_format)
        textTimes.append(textTime)
        # Tiến đến ngày tiếp theo
        current_date += timedelta(days=1)
    return textTimes
# Ví dụ sử dụng hàm
# getTextTimes('01-08-2024', '04-08-2024')
def getTextTimeNow(date_format = '%d-%m-%Y'):
    now  = datetime.now()
    return now.strftime(date_format)