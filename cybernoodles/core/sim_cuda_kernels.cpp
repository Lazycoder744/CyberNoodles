#include <torch/extension.h>

#include <vector>

namespace {

void check_float_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

void check_bool_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kBool, name, " must be bool");
}

void check_vec3_2d(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == 3, name, " must have shape [N, 3]");
}

void check_vec3_3d(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() == 3 && tensor.size(2) == 3, name, " must have shape [N, M, 3]");
}

void check_bool_2d(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() == 2, name, " must have shape [N, M]");
}

void check_float_2d(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() == 2, name, " must have shape [N, M]");
}

void check_same_device(const torch::Tensor& lhs, const torch::Tensor& rhs, const char* lhs_name, const char* rhs_name) {
    TORCH_CHECK(lhs.device() == rhs.device(), lhs_name, " and ", rhs_name, " must be on the same device");
}

}  // namespace

void note_collision_metrics_out_cuda(
    torch::Tensor left_hilt_prev,
    torch::Tensor left_hilt_curr,
    torch::Tensor left_tip_prev,
    torch::Tensor left_tip_curr,
    torch::Tensor left_ctrl_prev,
    torch::Tensor left_ctrl_curr,
    torch::Tensor right_hilt_prev,
    torch::Tensor right_hilt_curr,
    torch::Tensor right_tip_prev,
    torch::Tensor right_tip_curr,
    torch::Tensor right_ctrl_prev,
    torch::Tensor right_ctrl_curr,
    torch::Tensor good_centers,
    torch::Tensor good_half,
    torch::Tensor note_pos,
    torch::Tensor bad_half,
    torch::Tensor assist_enabled,
    torch::Tensor left_good_out,
    torch::Tensor right_good_out,
    torch::Tensor left_bad_out,
    torch::Tensor right_bad_out,
    torch::Tensor dl_out,
    torch::Tensor dr_out
);

void note_collision_metrics_out(
    torch::Tensor left_hilt_prev,
    torch::Tensor left_hilt_curr,
    torch::Tensor left_tip_prev,
    torch::Tensor left_tip_curr,
    torch::Tensor left_ctrl_prev,
    torch::Tensor left_ctrl_curr,
    torch::Tensor right_hilt_prev,
    torch::Tensor right_hilt_curr,
    torch::Tensor right_tip_prev,
    torch::Tensor right_tip_curr,
    torch::Tensor right_ctrl_prev,
    torch::Tensor right_ctrl_curr,
    torch::Tensor good_centers,
    torch::Tensor good_half,
    torch::Tensor note_pos,
    torch::Tensor bad_half,
    torch::Tensor assist_enabled,
    torch::Tensor left_good_out,
    torch::Tensor right_good_out,
    torch::Tensor left_bad_out,
    torch::Tensor right_bad_out,
    torch::Tensor dl_out,
    torch::Tensor dr_out
) {
    auto check_vec = [&](const torch::Tensor& tensor, const char* name) {
        check_float_cuda_contiguous(tensor, name);
        check_vec3_2d(tensor, name);
    };

    auto check_grid = [&](const torch::Tensor& tensor, const char* name) {
        check_float_cuda_contiguous(tensor, name);
        check_vec3_3d(tensor, name);
    };

    check_vec(left_hilt_prev, "left_hilt_prev");
    check_vec(left_hilt_curr, "left_hilt_curr");
    check_vec(left_tip_prev, "left_tip_prev");
    check_vec(left_tip_curr, "left_tip_curr");
    check_vec(left_ctrl_prev, "left_ctrl_prev");
    check_vec(left_ctrl_curr, "left_ctrl_curr");
    check_vec(right_hilt_prev, "right_hilt_prev");
    check_vec(right_hilt_curr, "right_hilt_curr");
    check_vec(right_tip_prev, "right_tip_prev");
    check_vec(right_tip_curr, "right_tip_curr");
    check_vec(right_ctrl_prev, "right_ctrl_prev");
    check_vec(right_ctrl_curr, "right_ctrl_curr");

    check_grid(good_centers, "good_centers");
    check_grid(good_half, "good_half");
    check_grid(note_pos, "note_pos");
    check_grid(bad_half, "bad_half");

    check_bool_cuda_contiguous(assist_enabled, "assist_enabled");
    TORCH_CHECK(assist_enabled.dim() == 1, "assist_enabled must have shape [N]");

    check_bool_cuda_contiguous(left_good_out, "left_good_out");
    check_bool_cuda_contiguous(right_good_out, "right_good_out");
    check_bool_cuda_contiguous(left_bad_out, "left_bad_out");
    check_bool_cuda_contiguous(right_bad_out, "right_bad_out");
    check_float_cuda_contiguous(dl_out, "dl_out");
    check_float_cuda_contiguous(dr_out, "dr_out");
    check_bool_2d(left_good_out, "left_good_out");
    check_bool_2d(right_good_out, "right_good_out");
    check_bool_2d(left_bad_out, "left_bad_out");
    check_bool_2d(right_bad_out, "right_bad_out");
    check_float_2d(dl_out, "dl_out");
    check_float_2d(dr_out, "dr_out");

    check_same_device(left_hilt_prev, good_centers, "left_hilt_prev", "good_centers");
    check_same_device(left_hilt_prev, assist_enabled, "left_hilt_prev", "assist_enabled");
    check_same_device(left_hilt_prev, left_good_out, "left_hilt_prev", "left_good_out");

    const auto n = good_centers.size(0);
    const auto m = good_centers.size(1);

    auto check_nm = [&](const torch::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.size(0) == n && tensor.size(1) == m, name, " must match [N, M]");
    };

    TORCH_CHECK(left_hilt_prev.size(0) == n, "left_hilt_prev must have N rows");
    TORCH_CHECK(left_hilt_curr.size(0) == n, "left_hilt_curr must have N rows");
    TORCH_CHECK(left_tip_prev.size(0) == n, "left_tip_prev must have N rows");
    TORCH_CHECK(left_tip_curr.size(0) == n, "left_tip_curr must have N rows");
    TORCH_CHECK(left_ctrl_prev.size(0) == n, "left_ctrl_prev must have N rows");
    TORCH_CHECK(left_ctrl_curr.size(0) == n, "left_ctrl_curr must have N rows");
    TORCH_CHECK(right_hilt_prev.size(0) == n, "right_hilt_prev must have N rows");
    TORCH_CHECK(right_hilt_curr.size(0) == n, "right_hilt_curr must have N rows");
    TORCH_CHECK(right_tip_prev.size(0) == n, "right_tip_prev must have N rows");
    TORCH_CHECK(right_tip_curr.size(0) == n, "right_tip_curr must have N rows");
    TORCH_CHECK(right_ctrl_prev.size(0) == n, "right_ctrl_prev must have N rows");
    TORCH_CHECK(right_ctrl_curr.size(0) == n, "right_ctrl_curr must have N rows");
    TORCH_CHECK(assist_enabled.size(0) == n, "assist_enabled must have N elements");

    check_nm(good_half, "good_half");
    check_nm(note_pos, "note_pos");
    check_nm(bad_half, "bad_half");
    check_nm(left_good_out, "left_good_out");
    check_nm(right_good_out, "right_good_out");
    check_nm(left_bad_out, "left_bad_out");
    check_nm(right_bad_out, "right_bad_out");
    check_nm(dl_out, "dl_out");
    check_nm(dr_out, "dr_out");

    note_collision_metrics_out_cuda(
        left_hilt_prev,
        left_hilt_curr,
        left_tip_prev,
        left_tip_curr,
        left_ctrl_prev,
        left_ctrl_curr,
        right_hilt_prev,
        right_hilt_curr,
        right_tip_prev,
        right_tip_curr,
        right_ctrl_prev,
        right_ctrl_curr,
        good_centers,
        good_half,
        note_pos,
        bad_half,
        assist_enabled,
        left_good_out,
        right_good_out,
        left_bad_out,
        right_bad_out,
        dl_out,
        dr_out
    );
}

std::vector<torch::Tensor> note_collision_metrics(
    torch::Tensor left_hilt_prev,
    torch::Tensor left_hilt_curr,
    torch::Tensor left_tip_prev,
    torch::Tensor left_tip_curr,
    torch::Tensor left_ctrl_prev,
    torch::Tensor left_ctrl_curr,
    torch::Tensor right_hilt_prev,
    torch::Tensor right_hilt_curr,
    torch::Tensor right_tip_prev,
    torch::Tensor right_tip_curr,
    torch::Tensor right_ctrl_prev,
    torch::Tensor right_ctrl_curr,
    torch::Tensor good_centers,
    torch::Tensor good_half,
    torch::Tensor note_pos,
    torch::Tensor bad_half,
    torch::Tensor assist_enabled
) {
    auto bool_opts = torch::TensorOptions().device(good_centers.device()).dtype(torch::kBool);
    auto float_opts = torch::TensorOptions().device(good_centers.device()).dtype(torch::kFloat32);
    auto sizes = std::vector<int64_t>{good_centers.size(0), good_centers.size(1)};

    auto left_good_out = torch::empty(sizes, bool_opts);
    auto right_good_out = torch::empty(sizes, bool_opts);
    auto left_bad_out = torch::empty(sizes, bool_opts);
    auto right_bad_out = torch::empty(sizes, bool_opts);
    auto dl_out = torch::empty(sizes, float_opts);
    auto dr_out = torch::empty(sizes, float_opts);

    note_collision_metrics_out(
        left_hilt_prev,
        left_hilt_curr,
        left_tip_prev,
        left_tip_curr,
        left_ctrl_prev,
        left_ctrl_curr,
        right_hilt_prev,
        right_hilt_curr,
        right_tip_prev,
        right_tip_curr,
        right_ctrl_prev,
        right_ctrl_curr,
        good_centers,
        good_half,
        note_pos,
        bad_half,
        assist_enabled,
        left_good_out,
        right_good_out,
        left_bad_out,
        right_bad_out,
        dl_out,
        dr_out
    );

    return {left_good_out, right_good_out, left_bad_out, right_bad_out, dl_out, dr_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("note_collision_metrics", &note_collision_metrics, "Fused note collision metrics (CUDA)");
    m.def("note_collision_metrics_out", &note_collision_metrics_out, "Fused note collision metrics out variant (CUDA)");
}
