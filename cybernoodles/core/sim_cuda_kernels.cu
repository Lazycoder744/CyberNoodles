#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

struct Vec3 {
    float x;
    float y;
    float z;
};

__device__ __forceinline__ Vec3 make_vec3(float x, float y, float z) {
    return Vec3{x, y, z};
}

__device__ __forceinline__ Vec3 load_vec3(const float* base) {
    return make_vec3(base[0], base[1], base[2]);
}

__device__ __forceinline__ Vec3 add(Vec3 a, Vec3 b) {
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ Vec3 sub(Vec3 a, Vec3 b) {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ Vec3 mul(Vec3 a, float s) {
    return make_vec3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float length_sq(Vec3 a) {
    return dot(a, a);
}

__device__ __forceinline__ float point_segment_distance(Vec3 point, Vec3 seg_start, Vec3 seg_end) {
    const Vec3 seg = sub(seg_end, seg_start);
    const Vec3 rel = sub(point, seg_start);
    const float denom = fmaxf(length_sq(seg), 1.0e-6f);
    const float t = fminf(1.0f, fmaxf(0.0f, dot(rel, seg) / denom));
    const Vec3 closest = add(seg_start, mul(seg, t));
    const Vec3 delta = sub(point, closest);
    return sqrtf(length_sq(delta));
}

__device__ __forceinline__ bool segment_intersects_aabb(Vec3 seg_start, Vec3 seg_end, Vec3 box_center, Vec3 half_extents) {
    const Vec3 box_min = sub(box_center, half_extents);
    const Vec3 box_max = add(box_center, half_extents);
    const Vec3 delta = sub(seg_end, seg_start);

    float entry = 0.0f;
    float exit = 1.0f;

    const float p0[3] = {seg_start.x, seg_start.y, seg_start.z};
    const float d[3] = {delta.x, delta.y, delta.z};
    const float bmin[3] = {box_min.x, box_min.y, box_min.z};
    const float bmax[3] = {box_max.x, box_max.y, box_max.z};

    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(d[axis]) <= 1.0e-6f) {
            if (p0[axis] < bmin[axis] || p0[axis] > bmax[axis]) {
                return false;
            }
            continue;
        }

        float t0 = (bmin[axis] - p0[axis]) / d[axis];
        float t1 = (bmax[axis] - p0[axis]) / d[axis];
        if (t0 > t1) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        entry = fmaxf(entry, t0);
        exit = fminf(exit, t1);
        if (exit < entry) {
            return false;
        }
    }

    return exit >= 0.0f && entry <= 1.0f;
}

__global__ void note_collision_metrics_kernel(
    const float* left_hilt_prev,
    const float* left_hilt_curr,
    const float* left_tip_prev,
    const float* left_tip_curr,
    const float* left_ctrl_prev,
    const float* left_ctrl_curr,
    const float* right_hilt_prev,
    const float* right_hilt_curr,
    const float* right_tip_prev,
    const float* right_tip_curr,
    const float* right_ctrl_prev,
    const float* right_ctrl_curr,
    const float* good_centers,
    const float* good_half,
    const float* note_pos,
    const float* bad_half,
    const bool* assist_enabled,
    bool* left_good_out,
    bool* right_good_out,
    bool* left_bad_out,
    bool* right_bad_out,
    float* dl_out,
    float* dr_out,
    int64_t note_window,
    int64_t total_pairs
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) {
        return;
    }

    const int64_t env = idx / note_window;
    const int64_t env3 = env * 3;
    const int64_t pair3 = idx * 3;

    const Vec3 lhp = load_vec3(left_hilt_prev + env3);
    const Vec3 lhc = load_vec3(left_hilt_curr + env3);
    const Vec3 ltp = load_vec3(left_tip_prev + env3);
    const Vec3 ltc = load_vec3(left_tip_curr + env3);
    const Vec3 lcp = load_vec3(left_ctrl_prev + env3);
    const Vec3 lcc = load_vec3(left_ctrl_curr + env3);

    const Vec3 rhp = load_vec3(right_hilt_prev + env3);
    const Vec3 rhc = load_vec3(right_hilt_curr + env3);
    const Vec3 rtp = load_vec3(right_tip_prev + env3);
    const Vec3 rtc = load_vec3(right_tip_curr + env3);
    const Vec3 rcp = load_vec3(right_ctrl_prev + env3);
    const Vec3 rcc = load_vec3(right_ctrl_curr + env3);

    const Vec3 good_center = load_vec3(good_centers + pair3);
    const Vec3 good_half_extent = load_vec3(good_half + pair3);
    const Vec3 note_center = load_vec3(note_pos + pair3);
    const Vec3 bad_half_extent = load_vec3(bad_half + pair3);
    const bool assist = assist_enabled[env];

    bool left_good = segment_intersects_aabb(ltp, ltc, good_center, good_half_extent)
        || segment_intersects_aabb(lhp, lhc, good_center, good_half_extent)
        || segment_intersects_aabb(lhc, ltc, good_center, good_half_extent);
    bool right_good = segment_intersects_aabb(rtp, rtc, good_center, good_half_extent)
        || segment_intersects_aabb(rhp, rhc, good_center, good_half_extent)
        || segment_intersects_aabb(rhc, rtc, good_center, good_half_extent);

    if (assist) {
        left_good = left_good || segment_intersects_aabb(lcp, lcc, good_center, good_half_extent);
        right_good = right_good || segment_intersects_aabb(rcp, rcc, good_center, good_half_extent);
    }

    const bool left_bad = segment_intersects_aabb(ltp, ltc, note_center, bad_half_extent)
        || segment_intersects_aabb(lhp, lhc, note_center, bad_half_extent)
        || segment_intersects_aabb(lhc, ltc, note_center, bad_half_extent);
    const bool right_bad = segment_intersects_aabb(rtp, rtc, note_center, bad_half_extent)
        || segment_intersects_aabb(rhp, rhc, note_center, bad_half_extent)
        || segment_intersects_aabb(rhc, rtc, note_center, bad_half_extent);

    const float left_blade = point_segment_distance(note_center, lhc, ltc);
    const float left_tip_sweep = point_segment_distance(note_center, ltp, ltc);
    const float left_hilt_sweep = point_segment_distance(note_center, lhp, lhc);
    const float right_blade = point_segment_distance(note_center, rhc, rtc);
    const float right_tip_sweep = point_segment_distance(note_center, rtp, rtc);
    const float right_hilt_sweep = point_segment_distance(note_center, rhp, rhc);

    left_good_out[idx] = left_good;
    right_good_out[idx] = right_good;
    left_bad_out[idx] = left_bad;
    right_bad_out[idx] = right_bad;
    dl_out[idx] = fminf(left_blade, fminf(left_tip_sweep, left_hilt_sweep));
    dr_out[idx] = fminf(right_blade, fminf(right_tip_sweep, right_hilt_sweep));
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
) {
    const c10::cuda::CUDAGuard device_guard(left_hilt_prev.device());
    const int64_t note_window = good_centers.size(1);
    const int64_t total_pairs = good_centers.size(0) * note_window;

    if (total_pairs == 0) {
        return;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    note_collision_metrics_kernel<<<blocks, threads, 0, stream>>>(
        left_hilt_prev.data_ptr<float>(),
        left_hilt_curr.data_ptr<float>(),
        left_tip_prev.data_ptr<float>(),
        left_tip_curr.data_ptr<float>(),
        left_ctrl_prev.data_ptr<float>(),
        left_ctrl_curr.data_ptr<float>(),
        right_hilt_prev.data_ptr<float>(),
        right_hilt_curr.data_ptr<float>(),
        right_tip_prev.data_ptr<float>(),
        right_tip_curr.data_ptr<float>(),
        right_ctrl_prev.data_ptr<float>(),
        right_ctrl_curr.data_ptr<float>(),
        good_centers.data_ptr<float>(),
        good_half.data_ptr<float>(),
        note_pos.data_ptr<float>(),
        bad_half.data_ptr<float>(),
        assist_enabled.data_ptr<bool>(),
        left_good_out.data_ptr<bool>(),
        right_good_out.data_ptr<bool>(),
        left_bad_out.data_ptr<bool>(),
        right_bad_out.data_ptr<bool>(),
        dl_out.data_ptr<float>(),
        dr_out.data_ptr<float>(),
        note_window,
        total_pairs
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "note_collision_metrics_kernel launch failed: ", cudaGetErrorString(err));
}
