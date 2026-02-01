"""
Hệ thống nhận dạng ngôn ngữ ký hiệu dựa vào Hu's Moments
Phương pháp đánh giá: LOOCV (Leave-One-Out Cross Validation)

Luồng xử lý:
1. Đọc ảnh từ Img_input → chuyển thành binary → lưu vào Img_binary
2. Đọc ảnh từ Img_binary → tính Hu's moments → LOOCV
"""

import cv2
import numpy as np
import os
from pathlib import Path


def convert_to_binary(input_dir, binary_dir):
    """
    Chuyển tất cả ảnh từ Img_input sang ảnh binary và lưu vào Img_binary
    
    Args:
        input_dir: Thư mục chứa ảnh gốc (Img_input)
        binary_dir: Thư mục lưu ảnh binary (Img_binary)
    """
    input_path = Path(input_dir)
    binary_path = Path(binary_dir)
    
    print("\n[BƯỚC 1] Chuyển đổi ảnh sang binary...")
    print("-" * 50)
    
    count = 0
    # Duyệt qua các thư mục con (mỗi ký hiệu)
    for symbol_dir in sorted(input_path.iterdir()):
        if symbol_dir.is_dir():
            symbol_name = symbol_dir.name
            
            # Tạo thư mục tương ứng trong Img_binary
            output_symbol_dir = binary_path / symbol_name
            output_symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Duyệt qua các ảnh
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(symbol_dir.glob(ext))
            
            # Loại bỏ trùng lặp
            seen = set()
            unique_files = []
            for f in image_files:
                if f.name.lower() not in seen:
                    seen.add(f.name.lower())
                    unique_files.append(f)
            
            for img_file in sorted(unique_files, key=lambda x: x.name.lower()):
                try:
                    # Đọc ảnh gốc
                    img = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"   Lỗi đọc: {img_file.name}")
                        continue
                    
                    # Chuyển sang ảnh xám
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Nhị phân hóa (Otsu's thresholding)
                    # Tùy vào ảnh, có thể cần THRESH_BINARY hoặc THRESH_BINARY_INV
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Lưu ảnh binary
                    output_path = output_symbol_dir / img_file.name
                    is_success, im_buf_arr = cv2.imencode(".jpg", binary)
                    if is_success:
                        im_buf_arr.tofile(str(output_path))
                        print(f"   ✓ {symbol_name}/{img_file.name}")
                        count += 1
                        
                except Exception as e:
                    print(f"   Lỗi xử lý {img_file.name}: {e}")
    
    print(f"\n   → Đã chuyển đổi {count} ảnh sang binary")
    print(f"   → Lưu tại: {binary_path}")


def calculate_hu_moments(image_path):
    """
    Tính toán 7 Hu moments từ ảnh binary
    
    Args:
        image_path: Đường dẫn đến file ảnh binary
        
    Returns:
        numpy array chứa 7 Hu moments (đã log transform)
    """
    # Đọc ảnh binary (hỗ trợ đường dẫn Unicode)
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    # Đảm bảo ảnh là binary (chỉ có 2 giá trị 0 và 255)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Tính moments
    moments = cv2.moments(binary)
    
    # Tính Hu moments
    hu_moments = cv2.HuMoments(moments)
    
    # Log transform để giảm độ lớn của giá trị
    # Sử dụng -sign(h) * log10(|h|) để giữ dấu
    for i in range(7):
        if hu_moments[i] != 0:
            hu_moments[i] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]))
    
    return hu_moments.flatten()


def load_dataset(input_dir):
    """
    Load toàn bộ dataset từ thư mục ảnh binary
    
    Args:
        input_dir: Thư mục chứa ảnh binary (Img_binary)
        
    Returns:
        data: list các tuple (hu_moments, label, image_path)
        labels: list các tên ký hiệu
    """
    data = []
    labels = []
    
    input_path = Path(input_dir)
    
    print("\n[BƯỚC 2] Tính Hu's Moments từ ảnh binary...")
    print("-" * 50)
    
    # Duyệt qua các thư mục con
    for symbol_dir in sorted(input_path.iterdir()):
        if symbol_dir.is_dir():
            symbol_name = symbol_dir.name
            if symbol_name not in labels:
                labels.append(symbol_name)
            
            # Duyệt qua các ảnh trong thư mục
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(symbol_dir.glob(ext))
            
            # Loại bỏ trùng lặp
            seen = set()
            unique_files = []
            for f in image_files:
                if f.name.lower() not in seen:
                    seen.add(f.name.lower())
                    unique_files.append(f)
            
            for img_file in sorted(unique_files, key=lambda x: x.name.lower()):
                try:
                    hu = calculate_hu_moments(str(img_file))
                    data.append((hu, symbol_name, str(img_file)))
                    print(f"   ✓ {symbol_name}/{img_file.name}")
                except Exception as e:
                    print(f"   Lỗi {img_file.name}: {e}")
    
    return data, labels


def euclidean_distance(hu1, hu2):
    """
    Tính khoảng cách Euclidean giữa hai vector Hu moments
    """
    return np.sqrt(np.sum((hu1 - hu2) ** 2))


def classify_knn(test_hu, training_data, k=1):
    """
    Phân loại sử dụng k-NN
    
    Args:
        test_hu: Hu moments của mẫu cần phân loại
        training_data: List các tuple (hu_moments, label)
        k: Số láng giềng gần nhất
        
    Returns:
        Nhãn được dự đoán
    """
    distances = []
    
    for train_hu, label, _ in training_data:
        dist = euclidean_distance(test_hu, train_hu)
        distances.append((dist, label))
    
    # Sắp xếp theo khoảng cách
    distances.sort(key=lambda x: x[0])
    
    # Lấy k láng giềng gần nhất
    k_nearest = distances[:k]
    
    # Bình chọn
    votes = {}
    for _, label in k_nearest:
        votes[label] = votes.get(label, 0) + 1
    
    # Trả về nhãn có nhiều phiếu nhất
    return max(votes, key=votes.get)


def loocv_evaluate(data, k=1):
    """
    Đánh giá mô hình sử dụng Leave-One-Out Cross Validation
    
    Args:
        data: List các tuple (hu_moments, label, image_path)
        k: Tham số k cho k-NN
        
    Returns:
        accuracy: Độ chính xác
        results: Chi tiết kết quả
    """
    correct = 0
    total = len(data)
    results = []
    
    print(f"\n{'='*70}")
    print(f"ĐÁNH GIÁ LOOCV VỚI k={k}")
    print(f"{'='*70}")
    
    for i in range(total):
        # Lấy mẫu test
        test_hu, true_label, test_path = data[i]
        
        # Tạo tập training (tất cả mẫu trừ mẫu test)
        training_data = data[:i] + data[i+1:]
        
        # Phân loại
        predicted_label = classify_knn(test_hu, training_data, k)
        
        # Kiểm tra kết quả
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
        
        results.append({
            'test_path': test_path,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct': is_correct
        })
        
        # In kết quả từng mẫu
        status = "✓" if is_correct else "✗"
        img_name = Path(test_path).name
        symbol_name = Path(test_path).parent.name
        print(f"[{status}] {symbol_name}/{img_name}: Thực tế={true_label}, Dự đoán={predicted_label}")
    
    accuracy = correct / total * 100
    
    return accuracy, results


def create_confusion_matrix(results, labels):
    """
    Tạo ma trận nhầm lẫn
    """
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    for r in results:
        true_idx = label_to_idx[r['true_label']]
        pred_idx = label_to_idx[r['predicted_label']]
        matrix[true_idx][pred_idx] += 1
    
    return matrix


def print_confusion_matrix(matrix, labels):
    """
    In ma trận nhầm lẫn
    """
    print("\n" + "="*70)
    print("MA TRẬN NHẦM LẪN (Confusion Matrix)")
    print("="*70)
    
    # Header
    header = "Thực tế\\Dự đoán"
    print(f"{header:>15}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print()
    
    # Rows
    for i, label in enumerate(labels):
        print(f"{label:>15}", end="")
        for j in range(len(labels)):
            print(f"{matrix[i][j]:>10}", end="")
        print()


def calculate_metrics_per_class(matrix, labels):
    """
    Tính Precision, Recall, F1-score cho từng lớp
    """
    print("\n" + "="*70)
    print("CHỈ SỐ ĐÁNH GIÁ THEO TỪNG LỚP")
    print("="*70)
    print(f"{'Lớp':>10} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-"*56)
    
    metrics = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[:, i]) - tp
        fn = sum(matrix[i, :]) - tp
        support = sum(matrix[i, :])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'label': label,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })
        
        print(f"{label:>10} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f} {support:>10}")
    
    # Macro average
    avg_precision = np.mean([m['precision'] for m in metrics])
    avg_recall = np.mean([m['recall'] for m in metrics])
    avg_f1 = np.mean([m['f1'] for m in metrics])
    total_support = sum([m['support'] for m in metrics])
    
    print("-"*56)
    print(f"{'Macro Avg':>10} {avg_precision:>12.4f} {avg_recall:>12.4f} {avg_f1:>12.4f} {total_support:>10}")
    
    return metrics


def save_results(accuracy, results, labels, matrix, output_dir):
    """
    Lưu kết quả ra file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / 'results.txt'
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("KẾT QUẢ NHẬN DẠNG NGÔN NGỮ KÝ HIỆU\n")
        f.write("Phương pháp: Hu's Moments + LOOCV\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Số lượng ký hiệu: {len(labels)}\n")
        f.write(f"Các ký hiệu: {', '.join(labels)}\n")
        f.write(f"Tổng số mẫu: {len(results)}\n")
        f.write(f"Số mẫu mỗi ký hiệu: {len(results)//len(labels)}\n\n")
        
        f.write(f"ĐỘ CHÍNH XÁC: {accuracy:.2f}%\n\n")
        
        # Chi tiết từng mẫu
        f.write("CHI TIẾT KẾT QUẢ:\n")
        f.write("-"*70 + "\n")
        for r in results:
            status = "✓" if r['correct'] else "✗"
            img_name = Path(r['test_path']).name
            symbol = Path(r['test_path']).parent.name
            f.write(f"[{status}] {symbol}/{img_name}: Thực tế={r['true_label']}, Dự đoán={r['predicted_label']}\n")
        
        # Ma trận nhầm lẫn
        f.write("\n" + "="*70 + "\n")
        f.write("MA TRẬN NHẦM LẪN:\n")
        f.write("-"*70 + "\n")
        
        header = "Thực tế\\Dự đoán"
        f.write(f"{header:>15}")
        for label in labels:
            f.write(f"{label:>10}")
        f.write("\n")
        
        for i, label in enumerate(labels):
            f.write(f"{label:>15}")
            for j in range(len(labels)):
                f.write(f"{matrix[i][j]:>10}")
            f.write("\n")
    
    print(f"\nKết quả đã được lưu vào: {result_file}")


def main():
    """
    Hàm chính
    """
    # Đường dẫn
    base_dir = Path(__file__).parent
    input_dir = base_dir / 'Img_input'    # Thư mục ảnh gốc
    binary_dir = base_dir / 'Img_binary'  # Thư mục ảnh binary
    output_dir = base_dir / 'Output'
    
    print("="*70)
    print("HỆ THỐNG NHẬN DẠNG NGÔN NGỮ KÝ HIỆU")
    print("Dựa trên Hu's Moments và LOOCV")
    print("="*70)
    
    # Kiểm tra thư mục input
    if not input_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục {input_dir}")
        return
    
    # BƯỚC 1: Chuyển ảnh sang binary
    convert_to_binary(input_dir, binary_dir)
    
    # BƯỚC 2: Load dataset từ ảnh binary và tính Hu's moments
    data, labels = load_dataset(binary_dir)
    
    print(f"\n   - Số lượng ký hiệu: {len(labels)}")
    print(f"   - Các ký hiệu: {', '.join(labels)}")
    print(f"   - Tổng số mẫu: {len(data)}")
    if len(labels) > 0:
        print(f"   - Số mẫu mỗi ký hiệu: {len(data)//len(labels)}")
    
    if len(data) == 0:
        print("Lỗi: Không có dữ liệu để đánh giá!")
        return
    
    # BƯỚC 3: Đánh giá với LOOCV
    print("\n[BƯỚC 3] Đánh giá với LOOCV (k-NN, k=1)...")
    print("-" * 50)
    k = 1  # Sử dụng 1-NN
    accuracy, results = loocv_evaluate(data, k)
    
    # Tạo và in ma trận nhầm lẫn
    matrix = create_confusion_matrix(results, labels)
    print_confusion_matrix(matrix, labels)
    
    # Tính các chỉ số đánh giá
    calculate_metrics_per_class(matrix, labels)
    
    # In kết quả tổng hợp
    print("\n" + "="*70)
    print("KẾT QUẢ TỔNG HỢP")
    print("="*70)
    correct_count = sum(1 for r in results if r['correct'])
    print(f"Số mẫu đúng: {correct_count}/{len(results)}")
    print(f"ĐỘ CHÍNH XÁC: {accuracy:.2f}%")
    
    # Lưu kết quả
    save_results(accuracy, results, labels, matrix, output_dir)
    
    print("\n" + "="*70)
    print("HOÀN THÀNH!")
    print("="*70)


if __name__ == "__main__":
    main()
