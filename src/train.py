from ultralytics import YOLOv10

# Đường dẫn đến mô hình YOLO đã được huấn luyện
MODEL_PATH = "yolov10n.pt"
model = YOLOv10(MODEL_PATH)

# # Đường dẫn đến tệp YAML chứa thông tin về bộ dữ liệu
YAML_PATH = '../helmet-safety-detection/data_set/data.yaml'

# Số lượng epoch để huấn luyện mô hình
# Epoch là một lần lặp qua toàn bộ bộ dữ liệu huấn luyện.
# Khi số lượng epoch tăng, mô hình sẽ được huấn luyện trên bộ dữ liệu nhiều lần hơn,
# giúp mô hình học được nhiều hơn từ dữ liệu.
# Tuy nhiên, nếu số lượng epoch quá cao, mô hình có thể bị overfitting (quá khớp).
EPOCHS = 50

# Kích thước ảnh đầu vào
# IMG_SIZE là kích thước của ảnh đầu vào mà mô hình sẽ xử lý.
# Ảnh sẽ được resize về kích thước này trước khi đưa vào mô hình.
# Kích thước lớn hơn có thể giúp mô hình nhận diện tốt hơn nhưng cũng đòi hỏi nhiều tài nguyên tính toán hơn.
IMG_SIZE = 640

# Kích thước batch trong quá trình huấn luyện
# BATCH_SIZE là số lượng mẫu được xử lý trước khi mô hình cập nhật trọng số một lần.
# Batch size lớn hơn có thể giúp huấn luyện nhanh hơn và ổn định hơn, nhưng cũng yêu cầu nhiều bộ nhớ hơn.
# bạn có thể chuyển thành 256 nếu bạn có card màn hình đủ khỏe
BATCH_SIZE = 16

model.train(data = YAML_PATH,
            epochs = EPOCHS,
            batch = BATCH_SIZE,
            imgsz = IMG_SIZE)

# Đường dẫn đến mô hình đã được huấn luyện tốt nhất
TRAINED_MODEL_PATH = 'runs/detect/train2/weights/best.pt'
model = YOLOv10(TRAINED_MODEL_PATH)

# Đánh giá mô hình trên tập dữ liệu kiểm tra
model.val(data=YAML_PATH,
          imgsz=IMG_SIZE,
          split='test')


PATH_MODEL_DETECT = "runs/detect/train2/weights/best.pt"
model = YOLOv10(PATH_MODEL_DETECT)
result = model("data_set/test/images/helmet-12-_jpg.rf.e060b7cfb30b033ca50fc1b9ed48efe7.jpg")

for element in result:
    element.save()