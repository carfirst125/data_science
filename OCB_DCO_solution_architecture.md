# Solution Architecture — Ứng dụng AI sinh Banner tự động

> File này tổng hợp lại toàn bộ nội dung đã trao đổi để tiếp tục làm việc trên phiên/tài khoản Claude khác. Copy toàn bộ nội dung dưới đây và dán vào đầu cuộc trò chuyện mới kèm câu: *"Đây là bối cảnh dự án tôi đã trao đổi trước đó, tiếp tục hỗ trợ tôi từ đây."*

## 1. Bối cảnh dự án

Dự án: tạo banner marketing tự động theo kịch bản cho trước, dùng **GPT-image-1-mini** để sinh ảnh.

**Input của hệ thống:**
- Brief: yêu cầu tạo banner cho campaign nào, đối tượng khách hàng nào.
- Guideline: quy chuẩn thiết kế banner (font, màu, bố cục, **độ dài text** cho phép...).
- Image element bắt buộc (nếu có): logo, ảnh sản phẩm do user cung cấp.
- Text script: yêu cầu về thông điệp — app phải tự sinh câu chữ dựa trên script, tuân theo độ dài quy định trong guideline.

**Trên UI:** user upload 1 file YAML tổng hợp (yêu cầu, tên file brief, tên ảnh đính kèm) + toàn bộ ảnh liên quan. Nhấn **Run** để hệ thống xử lý sinh banner.

## 2. Quyết định kiến trúc quan trọng

### 2.1. Không dùng Databricks
Ban đầu có đề cập triển khai trên Databricks, nhưng sau khi phân tích: ứng dụng này là web app + orchestrator gọi API sinh ảnh, không có workload big data / Spark / training model — Databricks không cần thiết, chỉ tăng chi phí và độ phức tạp vận hành. **Quyết định: dùng thuần Azure**, đúng với stack CICD đã chọn (Azure DevOps, ACR, ACA).

### 2.2. Lựa chọn DB — 2 tầng lưu trữ
Dữ liệu cần lưu rất đa dạng: image element, guideline, thông điệp text, các thành phần phát sinh trong quá trình xử lý (ảnh + text).

| Loại dữ liệu | Nơi lưu | Lý do |
|---|---|---|
| File nhị phân: ảnh input/element, guideline gốc, banner sinh ra (draft + final) | **Azure Blob Storage (ADLS Gen2)** | Tối ưu cho object storage, chi phí thấp, versioned theo job |
| Dữ liệu có cấu trúc: brief, job status, text script, rule guideline đã parse, registry trỏ tới file trên Blob | **Azure Database for PostgreSQL Flexible Server** | Có quan hệ rõ ràng (brief → job → assets), cần ACID để track trạng thái job; nếu sau này cần semantic search trên guideline chỉ cần bật extension `pgvector`, không cần vector DB riêng |
| Hàng đợi job / trạng thái xử lý real-time (tuỳ chọn) | **Azure Cache for Redis** | Nếu thời gian sinh ảnh đủ lâu cần pattern async (submit → poll status) |

## 3. Hạ tầng (Terraform, provision qua Azure DevOps)

Resource nhóm theo chức năng:

**Bảo mật / Observability**
- Azure Key Vault — lưu API key GPT-image-1-mini, connection string DB
- Managed Identity cho ACA — truy cập Key Vault/Storage/ACR không cần hardcode secret
- Log Analytics Workspace + Application Insights — log & trace job xử lý

**Lưu trữ dữ liệu**
- ADLS Gen2 (Blob Storage)
- Azure Database for PostgreSQL
- Azure Cache for Redis (tuỳ chọn)

**Compute / Registry**
- ACR (Azure Container Registry)
- Container Apps Environment
- ACA — container `banner-app` chạy UI + API + orchestrator

## 4. CICD — 3 pipeline trên Azure DevOps

**Pipeline 1 · Infrastructure**
1. `terraform init`
2. `terraform plan`
3. Manual approval (cho môi trường prod)
4. `terraform apply`

**Pipeline 2 · Build image**
1. Checkout code
2. Unit test
3. `docker build`
4. Tag: git-sha + latest
5. Push → ACR (chỉ dùng ACR, bỏ DockerHub để tránh double registry và tận dụng Managed Identity pull image)

**Pipeline 3 · Deploy**
1. Trigger sau khi Pipeline 2 thành công
2. `az containerapp update`
3. Health check
4. Rollback nếu fail

## 5. App block diagram — luồng xử lý runtime chính

| # | Bước | Mô tả |
|---|---|---|
| ① | User | Upload YAML (brief, tên file, tên ảnh) + ảnh đính kèm |
| ② | UI + Backend API (trong ACA) | Parse YAML, validate |
| ③ | Lưu dữ liệu | Ảnh → Blob · Metadata/text → Postgres |
| ④ | Main Orchestrator | (User nhấn Run) Đọc brief + guideline rule (độ dài text, quy chuẩn) từ Postgres; lấy ảnh bắt buộc từ Blob |
| ⑤ | Sinh thông điệp text | LLM text (vd Azure OpenAI GPT) theo text script + giới hạn độ dài guideline |
| ⑥ | Gọi GPT-image-1-mini | Sinh ảnh banner từ prompt + ảnh element tham chiếu |
| ⑦ | Composite text (Pillow) | Overlay text chính xác lên ảnh — model ảnh không đảm bảo đúng ký tự nên text luôn render bằng code, không phải AI vẽ |
| ⑧ | Lưu kết quả | Banner final/draft → Blob · cập nhật job status → Postgres |
| ⑨ | UI hiển thị | Preview + link tải banner cho user |

**Lưu ý thiết kế quan trọng:** GPT-image-1-mini không đảm bảo render đúng text/ký tự theo guideline độ dài, nên tách hẳn bước sinh text (LLM riêng, bước ⑤) và composite text lên ảnh (bước ⑦) thay vì để model ảnh tự viết chữ trong ảnh.

## 6. Vòng lặp Repair / Feedback (nối tiếp bước ⑨)

User xem preview, nhập feedback tự nhiên (vd: "font chữ chưa đúng quy chuẩn", "nhân vật đưa hai tay lên trời, cười tươi hơn"), nhấn **Repair**. Lặp lại đến khi Approve.

| # | Bước | Mô tả |
|---|---|---|
| ⑨ | Preview + ô feedback | User gõ góp ý tự nhiên, nhấn "Repair" |
| ⑩ | Feedback Interpreter (LLM) | Phân loại: lỗi text/font · lỗi hình ảnh · cả hai |
| ⑪a | Sửa text/font | Re-composite (Pillow) đúng font/size theo guideline — **không gọi lại model ảnh** |
| ⑪b | Sửa hình ảnh | GPT-image-1-mini edit: ảnh hiện tại + prompt mô tả thay đổi (dáng, biểu cảm...) |
| ⑫ | Lưu version mới | Blob: cùng container job, thêm subfolder v2, v3... · Postgres: thêm row liên kết `parent_version_id` |
| ↺ | Quay lại ⑨ | Hiển thị preview mới, lặp lại đến khi user nhấn Approve |

### Vì sao sửa text không phải lúc nào cũng gọi GPT-image-1-mini
Vì text **không bao giờ** do model ảnh vẽ ra pixel — luôn là layer riêng do code render (xem mục 7). Nên có 2 case, cả hai đều **không** đụng tới model ảnh:
1. Sai font/size/vị trí → chỉ re-render layer text bằng code theo guideline.
2. Đổi nội dung câu chữ → gọi LLM viết lại text mới, vẫn render bằng code lên layer đó, ảnh nền giữ nguyên.

GPT-image-1-mini chỉ được gọi khi feedback động tới phần **hình ảnh** (dáng nhân vật, bối cảnh, màu sắc, bố cục đồ hoạ).

### Data model versioning
Ảnh cũ và mới lưu chung 1 container theo `job_id`, phân biệt bằng version — không ghi đè, không tách container riêng.

| Trường | Ý nghĩa |
|---|---|
| `job_id` | Container/khoá gốc — mọi version của cùng banner nằm chung đây |
| `version_no` | 1, 2, 3... tăng dần mỗi lần Repair |
| `parent_version_id` | Trỏ về version vừa bị sửa — dựng được cây lịch sử chỉnh sửa |
| `feedback_text` | Nguyên văn góp ý user nhập ở version đó |
| `edit_type` | `text_fix` / `visual_edit` / `both` |
| `image_blob_path` | `generated/{job_id}/v{n}/banner.png` |
| `status` | `draft` / `approved` — chỉ 1 version approved mỗi job |

## 7. Thiết kế layer — banner là manifest nhiều layer, không phải 1 ảnh flat

Thay vì sinh 1 ảnh flat rồi tách lớp sau (rủi ro, dễ lỗi viền), tách layer ngay từ lúc generate. Z-order từ dưới lên trên:

| Layer | Nguồn | Ghi chú |
|---|---|---|
| Background / bối cảnh | AI (GPT-image-1-mini) | Sinh riêng, không chứa nhân vật lẫn text |
| Nhân vật / chủ thể | AI + tách nền | GPT-image-1-mini sinh + matting (rembg/SAM) → PNG có alpha |
| Logo / sản phẩm bắt buộc | User upload | Giữ nguyên xuyên suốt, không qua AI |
| Text (headline, CTA, body copy) | Code render | Vector, font/size theo guideline — không phải pixel do AI vẽ |

Mỗi layer lưu file riêng (PNG có alpha, hoặc vector) + toạ độ/kích thước/blend mode trong `manifest.json`.

### Lợi ích: repair chọn lọc đúng layer

| Feedback ví dụ | Layer bị sửa | Gọi GPT-image-1-mini? |
|---|---|---|
| "Font chữ chưa đúng quy chuẩn" | Text | Không |
| "Đổi câu chữ thông điệp" | Text | Không (chỉ gọi LLM text) |
| "Nhân vật đưa 2 tay lên trời, cười tươi hơn" | Nhân vật | Có — chỉ sinh lại layer này |
| "Đổi tông màu/bối cảnh phía sau" | Background | Có — chỉ sinh lại layer này |

Vì mỗi layer là 1 lệnh gọi riêng, repair không bao giờ vô tình làm hỏng logo hay đổi lại toàn bộ bố cục — khác với sửa trên 1 ảnh flat.

### Bàn giao cho designer

```
Manifest + layers (Blob)
  → Export to Figma (Figma API): mỗi layer thành 1 node riêng,
    text layer → Figma text node thật, sửa được font/màu trực tiếp
  → Designer chỉnh tay: di chuyển, resize, đổi màu từng object trong Figma
```

Không dùng PSD vì không có thư viện server-side đáng tin cậy để ghi layer PSD; Figma API tạo node theo layer trực tiếp (headless), và designer thường đã làm việc trên Figma.

## 8. Việc còn mở / cần quyết định tiếp

- Chọn LLM cụ thể cho bước sinh text (⑤) và Feedback Interpreter (⑩) — Azure OpenAI hay provider khác.
- Chi tiết matting/segmentation cho layer nhân vật (rembg vs SAM vs API có sẵn của GPT-image).
- Cơ chế async job cụ thể (có cần Redis queue + worker riêng hay xử lý đồng bộ trong ACA là đủ ở giai đoạn MVP).
- Thiết kế chi tiết bảng `banner_versions` / `banner_layers` trong Postgres (migration, index).
- Networking: có cần VNet + Private Endpoint cho Postgres/Storage hay để public endpoint + firewall rule ở giai đoạn đầu.
