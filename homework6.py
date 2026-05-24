import taichi as ti
import math

# ============================================================
# 1. 初始化 Taichi
# ============================================================

ti.init(arch=ti.cpu)

# 渲染分辨率
res = 256

# ============================================================
# 2. Buffer 定义
# ============================================================

# 左侧 Ground Truth 奶牛剪影
target_pixels = ti.field(dtype=ti.f32, shape=(res, res))

# 右侧当前优化结果
current_pixels = ti.field(dtype=ti.f32, shape=(res, res))

# GUI 显示：左 target，右 current
display_pixels = ti.field(dtype=ti.f32, shape=(res * 2, res))

# Loss，需要梯度
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

# 可优化参数
# 0  body_cx
# 1  body_cy
# 2  body_rx
# 3  body_ry
# 4  head_cx
# 5  head_cy
# 6  head_rx
# 7  head_ry
# 8  ear_size
# 9  horn_size
# 10 leg_width
# 11 leg_height
# 12 neck_width
# 13 body_taper
params = ti.field(dtype=ti.f32, shape=14, needs_grad=True)

# 目标奶牛参数，不需要梯度
target_params = ti.field(dtype=ti.f32, shape=14)

# 边缘柔化程度，越大边缘越硬
sharpness = 90.0


# ============================================================
# 3. SDF 基础函数
# ============================================================

@ti.func
def ellipse_sdf(x, y, cx, cy, rx, ry):
    qx = (x - cx) / rx
    qy = (y - cy) / ry
    return ti.sqrt(qx * qx + qy * qy) - 1.0


@ti.func
def box_sdf(x, y, cx, cy, sx, sy):
    qx = ti.abs(x - cx) - sx
    qy = ti.abs(y - cy) - sy

    ox = ti.max(qx, 0.0)
    oy = ti.max(qy, 0.0)

    outside = ti.sqrt(ox * ox + oy * oy)
    inside = ti.min(ti.max(qx, qy), 0.0)

    return outside + inside


@ti.func
def smooth_union(d1, d2, k):
    h = ti.max(0.0, ti.min(1.0, 0.5 + 0.5 * (d2 - d1) / k))
    return d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)


@ti.func
def smooth_subtract(d1, d2, k):
    # 从 d1 中减去 d2
    h = ti.max(0.0, ti.min(1.0, 0.5 - 0.5 * (d2 + d1) / k))
    return d1 * (1.0 - h) + (-d2) * h + k * h * (1.0 - h)


@ti.func
def sdf_to_alpha(d):
    return 1.0 / (1.0 + ti.exp(sharpness * d))


# ============================================================
# 4. 奶牛剪影 SDF
# ============================================================

@ti.func
def cow_sdf(
    x, y,
    body_cx, body_cy, body_rx, body_ry,
    head_cx, head_cy, head_rx, head_ry,
    ear_size, horn_size,
    leg_width, leg_height,
    neck_width, body_taper
):
    # --------------------------------------------------------
    # 身体：用两个椭圆叠加，形成上窄下宽的正面牛身体
    # --------------------------------------------------------

    upper_body = ellipse_sdf(
        x, y,
        body_cx,
        body_cy + 0.055,
        body_rx * (0.82 + body_taper),
        body_ry * 0.78
    )

    lower_body = ellipse_sdf(
        x, y,
        body_cx,
        body_cy - 0.075,
        body_rx,
        body_ry * 0.92
    )

    d_body = smooth_union(upper_body, lower_body, 0.08)

    # --------------------------------------------------------
    # 头部：上方偏圆的牛头
    # --------------------------------------------------------

    d_head = ellipse_sdf(
        x, y,
        head_cx,
        head_cy,
        head_rx,
        head_ry
    )

    # --------------------------------------------------------
    # 脖子：连接头和身体
    # --------------------------------------------------------

    d_neck = box_sdf(
        x, y,
        body_cx,
        (body_cy + head_cy) * 0.5 - 0.02,
        neck_width,
        0.105
    )

    # --------------------------------------------------------
    # 左右耳朵：横向椭圆
    # --------------------------------------------------------

    d_ear_l = ellipse_sdf(
        x, y,
        head_cx - head_rx * 0.92,
        head_cy + head_ry * 0.22,
        ear_size * 1.25,
        ear_size * 0.62
    )

    d_ear_r = ellipse_sdf(
        x, y,
        head_cx + head_rx * 0.92,
        head_cy + head_ry * 0.22,
        ear_size * 1.25,
        ear_size * 0.62
    )

    # --------------------------------------------------------
    # 左右牛角：上方小圆角凸起
    # --------------------------------------------------------

    d_horn_l = ellipse_sdf(
        x, y,
        head_cx - head_rx * 0.45,
        head_cy + head_ry * 0.92,
        horn_size * 0.65,
        horn_size * 1.05
    )

    d_horn_r = ellipse_sdf(
        x, y,
        head_cx + head_rx * 0.45,
        head_cy + head_ry * 0.92,
        horn_size * 0.65,
        horn_size * 1.05
    )

    # --------------------------------------------------------
    # 两条腿：参考图中是正面两条腿，中间有缺口
    # --------------------------------------------------------

    leg_y = body_cy - body_ry * 0.92

    d_leg_l = box_sdf(
        x, y,
        body_cx - body_rx * 0.38,
        leg_y - leg_height * 0.42,
        leg_width,
        leg_height
    )

    d_leg_r = box_sdf(
        x, y,
        body_cx + body_rx * 0.38,
        leg_y - leg_height * 0.42,
        leg_width,
        leg_height
    )

    # --------------------------------------------------------
    # 合并所有部件
    # --------------------------------------------------------

    d = d_body
    d = smooth_union(d, d_head, 0.075)
    d = smooth_union(d, d_neck, 0.06)
    d = smooth_union(d, d_ear_l, 0.045)
    d = smooth_union(d, d_ear_r, 0.045)
    d = smooth_union(d, d_horn_l, 0.04)
    d = smooth_union(d, d_horn_r, 0.04)
    d = smooth_union(d, d_leg_l, 0.05)
    d = smooth_union(d, d_leg_r, 0.05)

    # --------------------------------------------------------
    # 腿中间缺口
    # --------------------------------------------------------

    gap = box_sdf(
        x, y,
        body_cx,
        leg_y - leg_height * 0.50,
        body_rx * 0.18,
        leg_height * 0.82
    )

    d = smooth_subtract(d, gap, 0.035)

    return d


@ti.func
def target_alpha(x, y):
    d = cow_sdf(
        x, y,
        target_params[0],
        target_params[1],
        target_params[2],
        target_params[3],
        target_params[4],
        target_params[5],
        target_params[6],
        target_params[7],
        target_params[8],
        target_params[9],
        target_params[10],
        target_params[11],
        target_params[12],
        target_params[13],
    )
    return sdf_to_alpha(d)


@ti.func
def current_alpha(x, y):
    d = cow_sdf(
        x, y,
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
        params[7],
        params[8],
        params[9],
        params[10],
        params[11],
        params[12],
        params[13],
    )
    return sdf_to_alpha(d)


# ============================================================
# 5. 初始化目标参数和初始参数
# ============================================================

@ti.kernel
def init_params():
    # --------------------------------------------------------
    # 目标奶牛，接近你给的参考图
    # --------------------------------------------------------

    target_params[0] = 0.50    # body cx
    target_params[1] = 0.47    # body cy
    target_params[2] = 0.145   # body rx
    target_params[3] = 0.255   # body ry

    target_params[4] = 0.50    # head cx
    target_params[5] = 0.685   # head cy
    target_params[6] = 0.128   # head rx
    target_params[7] = 0.112   # head ry

    target_params[8] = 0.055   # ear size
    target_params[9] = 0.046   # horn size
    target_params[10] = 0.048  # leg width
    target_params[11] = 0.118  # leg height
    target_params[12] = 0.086  # neck width
    target_params[13] = 0.05   # body taper

    # --------------------------------------------------------
    # 初始优化形状，故意设得像右图那样不完美
    # --------------------------------------------------------

    params[0] = 0.50
    params[1] = 0.465
    params[2] = 0.165
    params[3] = 0.235

    params[4] = 0.50
    params[5] = 0.675
    params[6] = 0.105
    params[7] = 0.085

    params[8] = 0.030
    params[9] = 0.030
    params[10] = 0.040
    params[11] = 0.085
    params[12] = 0.060
    params[13] = -0.02


# ============================================================
# 6. 生成 Ground Truth 奶牛剪影
# ============================================================

@ti.kernel
def generate_target():
    for i, j in target_pixels:
        x = (i + 0.5) / res
        y = (j + 0.5) / res

        target_pixels[i, j] = target_alpha(x, y)


# ============================================================
# 7. 可微 Loss 计算
# ============================================================

@ti.kernel
def compute_loss():
    for i, j in target_pixels:
        x = (i + 0.5) / res
        y = (j + 0.5) / res

        pred = current_alpha(x, y)
        gt = target_pixels[i, j]

        diff = pred - gt
        loss[None] += diff * diff / (res * res)


# ============================================================
# 8. 非可微显示渲染
# ============================================================

@ti.kernel
def clear_display():
    for i, j in display_pixels:
        display_pixels[i, j] = 0.0


@ti.kernel
def render_current():
    for i, j in current_pixels:
        x = (i + 0.5) / res
        y = (j + 0.5) / res

        current_pixels[i, j] = current_alpha(x, y)


@ti.kernel
def render_display():
    for i, j in target_pixels:
        # 左边：目标剪影
        display_pixels[i, j] = target_pixels[i, j]

        # 右边：当前优化结果
        display_pixels[i + res, j] = current_pixels[i, j]


# ============================================================
# 9. 参数限制，防止优化跑飞
# ============================================================

def clamp_params():
    params[0] = max(0.42, min(0.58, params[0]))
    params[1] = max(0.35, min(0.58, params[1]))
    params[2] = max(0.08, min(0.24, params[2]))
    params[3] = max(0.15, min(0.34, params[3]))

    params[4] = max(0.42, min(0.58, params[4]))
    params[5] = max(0.58, min(0.78, params[5]))
    params[6] = max(0.06, min(0.18, params[6]))
    params[7] = max(0.06, min(0.16, params[7]))

    params[8] = max(0.015, min(0.085, params[8]))
    params[9] = max(0.015, min(0.075, params[9]))
    params[10] = max(0.025, min(0.075, params[10]))
    params[11] = max(0.05, min(0.16, params[11]))
    params[12] = max(0.035, min(0.12, params[12]))
    params[13] = max(-0.08, min(0.10, params[13]))


# ============================================================
# 10. 主程序
# ============================================================

def main():
    init_params()
    generate_target()

    # Adam 参数
    m = [0.0 for _ in range(14)]
    v = [0.0 for _ in range(14)]

    beta1 = 0.9
    beta2 = 0.999
    lr = 0.012
    eps = 1e-8

    gui = ti.GUI(
        "Ground Truth Silhouette  |  Optimizing Cow Silhouette",
        res=(res * 2, res),
    )

    print("=" * 70)
    print("Taichi Differentiable Cow Silhouette Optimization")
    print("=" * 70)

    total_iters = 300

    for iter in range(1, total_iters + 1):
        # ----------------------------------------------------
        # 清空 loss 和梯度
        # ----------------------------------------------------

        loss[None] = 0.0

        for k in range(14):
            params.grad[k] = 0.0

        # ----------------------------------------------------
        # 自动微分：只计算 loss，不做显示
        # ----------------------------------------------------

        with ti.ad.Tape(loss=loss):
            compute_loss()

        # ----------------------------------------------------
        # Adam 更新
        # ----------------------------------------------------

        for k in range(14):
            g = params.grad[k]

            m[k] = beta1 * m[k] + (1.0 - beta1) * g
            v[k] = beta2 * v[k] + (1.0 - beta2) * g * g

            m_hat = m[k] / (1.0 - beta1 ** iter)
            v_hat = v[k] / (1.0 - beta2 ** iter)

            params[k] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        clamp_params()

        # ----------------------------------------------------
        # 非可微显示渲染
        # 每帧都清空 display，避免重影
        # ----------------------------------------------------

        clear_display()
        render_current()
        render_display()

        # ----------------------------------------------------
        # 日志
        # ----------------------------------------------------

        if iter % 10 == 0 or iter == 1:
            print(
                f"迭代步数: {iter:03d}/{total_iters} | "
                f"总 Loss: {loss[None]:.5f} | "
                f"body_rx: {params[2]:.4f} | "
                f"body_ry: {params[3]:.4f} | "
                f"head_rx: {params[6]:.4f} | "
                f"head_ry: {params[7]:.4f}"
            )

        gui.set_image(display_pixels)
        gui.show()

    print("-" * 70)
    print("Optimization Finished")
    print("Final Parameters:")
    for k in range(14):
        print(f"params[{k}] = {params[k]:.5f}")
    print("=" * 70)


if __name__ == "__main__":
    main()