# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Hai anh em **Thành viên:** Đào Quang Cảnh (MSV 2A202602542), Trần Cao Quốc Định (MSV 2A202602939)

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

> Cột quan sát lấy từ **một frame giữa mỗi video** (bản bytetrack đặt cạnh bản botsort), chưa xem cả đoạn video, nên chưa nói được về chuyện đổi ID theo thời gian. Số hiệu ID hiện trên video không bắt đầu từ 1 (ví dụ khoảng 320–335 ở `video_4` bản bytetrack), chưa rõ nguyên nhân, nên không dùng số hiệu ID để đánh giá đổi ID.

## 1. Cấu hình đã chọn

Mỗi video: tracker bạn nộp, `conf`, `iou`, điều bạn **nhìn thấy** trên video, và một cấu hình đã thử rồi loại.

Nhóm chọn `botsort` với `conf` 0.3 và `iou` 0.5 cho cả năm video. Với `video_1` lựa chọn này dựa trên điểm số (mục 2). Với `video_2`–`video_5` không có nhãn nên lựa chọn chưa được kiểm chứng bằng điểm số.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.3 | 0.5 | Một frame giữa video (botsort): hộp quanh khoảng 7 người ở gần và giữa cảnh. Nhóm người nhỏ ở phía xa cuối quảng trường, hai trẻ đi xe đạp và người bị cắt ở mép ảnh không có hộp, khớp với số liệu: chỉ bắt được khoảng 22% số người trong nhãn, độ chính xác 92,3% | `bytetrack` 0.3/0.5 (HOTA 26,91); `botsort` `conf` 0.15 và 0.5; `iou` 0.4 và 0.7. Loại `iou` 0.7 dù HOTA cao hơn 0,5 điểm vì hộp giả 611 so với 337 và MOTA thấp hơn |
| video_2 (phố đêm, tĩnh, rất đông) | botsort | 0.3 | 0.5 | Một frame giữa video: hai bản đều đóng hộp nhóm người đi gần camera ở phía dưới; botsort có thêm 3 hộp ở phía trên bên phải mà bytetrack không có. Đám đông dày ở phía trên bên trái không bản nào đóng hộp. Số liệu: 1050/1050 frame có track, 67 ID, 11,7 hộp/frame | `bytetrack` 0.3/0.5 (47 ID, 9,4 hộp/frame; không có nhãn nên chưa chấm điểm) |
| video_3 (camera di động, ảnh nhỏ) | botsort | 0.3 | 0.5 | Một frame giữa video: ảnh nhỏ, mờ, camera di động; hai bản đóng hộp cùng 4 người (hai người đi phía trước, một người ở xa, một người lớn ở mép phải), chỉ khác số hiệu ID. Số liệu: 837/837 frame, 167 ID, 5,7 hộp/frame | `bytetrack` 0.3/0.5 (127 ID, 5,1 hộp/frame; chưa chấm điểm) |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.3 | 0.5 | Một frame giữa video (trong trung tâm thương mại, sàn bóng phản chiếu): hai bản đóng hộp những người đi phía trước với số hộp xấp xỉ nhau; không thấy hộp nào đóng vào hình phản chiếu trên sàn hay kính ở frame này. Số liệu: 900/900 frame, 73 ID, 6,9 hộp/frame | `bytetrack` 0.3/0.5 (61 ID, 6,4 hộp/frame; chưa chấm điểm) |
| video_5 (trên xe bus, giao lộ đông) | botsort | 0.3 | 0.5 | Một frame giữa video (camera trên xe bus): hai bản đều đóng hộp những người đang qua đường ở phía trước; botsort có thêm một hộp ở người nhỏ phía xa bên phải. Nhóm người xa ở giữa cảnh không bản nào đóng hộp. Số liệu: 747/750 frame có track, 77 ID, 4,1 hộp/frame | `bytetrack` 0.3/0.5 (710/750 frame có track, 62 ID, 2,7 hộp/frame; chưa chấm điểm) |

## 2. Số liệu video_1

Bảng HOTA / MOTA / IDF1 do `scripts/evaluate_practice.py` (TrackEval) ghi cho cấu hình nộp: `botsort`, `conf` 0.3, `iou` 0.5, đủ 600 frame. File `video_1.txt` nộp giống hệt file đã chấm.

```
HOTA DetA AssA DetRe DetPr AssRe AssPr LocA OWTA HOTA(0) LocA(0) HOTALocA(0) MOTA MOTP MODA CLR_Re CLR_Pr MTR PTR MLR CLR_TP CLR_FN CLR_FP IDSW MT PT ML Frag sMOTA IDF1 IDR IDP IDTP IDFN IDFP Dets GT_Dets IDs GT_IDs
29.46 18.095 48.223 18.596 78.888 51.153 82.992 83.666 29.899 35.813 78.45 28.095 19.811 81.471 19.945 21.759 92.306 12.903 19.355 67.742 4043 14538 337 25 8 12 42 99 15.779 29.354 18.137 76.941 3370 15211 1010 4380 18581 52 62
```

So sánh các cấu hình đã chấm trên `video_1` (mỗi lần chỉ đổi một tham số; `bytetrack` chỉ chấm một cấu hình):

| Cấu hình | HOTA | DetA | AssA | MOTA | IDF1 | Đổi ID | Hộp giả | Recall |
|---|---|---|---|---|---|---|---|---|
| bytetrack, conf 0.3, iou 0.5 | 26,91 | 15,07 | 48,13 | 17,29 | 25,71 | 12 | 107 | 17,9% |
| botsort, conf 0.15, iou 0.5 | 29,34 | 19,24 | 45,11 | 20,73 | 29,56 | 27 | 505 | 23,6% |
| **botsort, conf 0.3, iou 0.5 (nộp)** | **29,46** | 18,09 | 48,22 | 19,81 | 29,35 | 25 | 337 | 21,8% |
| botsort, conf 0.5, iou 0.5 | 27,17 | 14,30 | 51,65 | 15,25 | 24,56 | 10 | 229 | 16,5% |
| botsort, conf 0.3, iou 0.4 | 29,32 | 17,36 | 49,63 | 19,47 | 29,82 | 19 | 195 | 20,6% |
| botsort, conf 0.3, iou 0.7 | 29,97 | 18,41 | 49,06 | 19,02 | 29,70 | 33 | 611 | 22,5% |

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

## 3. Phân tích

**video_1 (đánh giá bằng số liệu).**
- Ở cùng `conf` 0.3 / `iou` 0.5, `botsort` hơn `bytetrack` ở HOTA (29,46 so với 26,91), MOTA (19,81 so với 17,29) và IDF1 (29,35 so với 25,71). Nhưng AssA gần như bằng nhau (48,22 so với 48,13), nên phần hơn đến từ phát hiện được nhiều người hơn (DetA 18,09 so với 15,07; recall 21,8% so với 17,9%), không phải giữ ID tốt hơn. Đổi lại `botsort` đổi ID nhiều hơn (25 so với 12 lần) và có nhiều hộp giả hơn (337 so với 107).
- Cảnh tĩnh, ban ngày, mật độ vừa nên `bytetrack` (chỉ dùng chuyển động) đã giữ ID ngang `botsort` (có Re-ID). Đây là suy luận từ số liệu, chưa kiểm chứng bằng xem video.
- Điểm nghẽn là recall: ở mọi cấu hình đã thử (`conf` 0.15–0.5, `iou` 0.4–0.7) recall chỉ khoảng 17–24%, và việc đổi `conf`, `iou` chỉ dịch HOTA dưới 3 điểm. Nhóm đoán nguyên nhân là detector cố định ở 640 px trên ảnh lớn nên bỏ sót người nhỏ, nhưng chưa kiểm tra trực tiếp.
- Giới hạn: chỉ một video, một cặp tracker, và `bytetrack` chưa được quét `conf`, nên chưa kết luận chắc tracker nào hợp hơn nói chung.

**video_5 (trên xe bus, rung lắc; đánh giá bằng số liệu từ file kết quả, chưa phải xem video).**
- Nhóm chọn `botsort`. So với `bytetrack` cùng `conf` 0.3 / `iou` 0.5, `botsort` có track ở 747/750 frame (`bytetrack`: 710/750, tức 40 frame trống), 4,06 hộp/frame (so với 2,70) và 77 ID (so với 62).
- Giả thuyết của nhóm: camera trên xe bus rung lắc nên vị trí giữa các frame khó đoán, tracker chỉ dựa vào chuyển động dễ mất người, còn Re-ID giúp nhận lại người sau khi bị mất. Số liệu phù hợp với giả thuyết (ít frame trống hơn) nhưng chưa chứng minh được.
- Không có nhãn nên chưa biết phần hộp tăng thêm là người thật hay hộp giả, và 15 ID tăng thêm là bắt thêm người hay đổi ID nhiều hơn. Kết quả `video_1` cho thấy `botsort` có thể bắt thêm người nhưng đổi ID nhiều hơn, nên cần xem video để xác nhận.

## 4. Nếu có thêm thời gian

Xem cả đoạn video `video_2`–`video_5` ở hai bản (bytetrack và botsort), không chỉ một frame, để kiểm tra đổi ID theo thời gian và xác nhận lựa chọn botsort; quét `conf` cho `bytetrack` để so công bằng với `botsort`; xem các frame `video_1` mà người nhỏ bị bỏ sót để kiểm tra giả thuyết về detector 640 px.

## 5. Phân công hoàn thiện báo cáo

Hai thành viên chia đều phần còn lại là xem video và xác nhận nội dung báo cáo, mỗi người chịu trách nhiệm đúng phần của mình:

| Thành viên | Việc phụ trách |
|---|---|
| Đào Quang Cảnh | Xem cả đoạn `video_1`, `video_2`, `video_3` (bản bytetrack và botsort), xác nhận hoặc sửa cột quan sát; đối chiếu bảng số liệu mục 2 với file kết quả |
| Trần Cao Quốc Định | Xem cả đoạn `video_4`, `video_5` (bản bytetrack và botsort), xác nhận hoặc sửa cột quan sát; đọc lại mục 3 và mục 4, kiểm tra tên, MSV và file nộp trước khi nộp |
