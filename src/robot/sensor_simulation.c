/**
 * @file sensor_simulation.c
 * @brief 仿真传感器深度实现
 * 
 * 实现激光雷达仿真、深度相机仿真、IMU仿真、射线求交。
 * 基于内部纯C物理引擎（SimPhysicsPipeline）的碰撞物体进行真实物理射线追踪。
 * 
 * 数据来源：内部纯C物理引擎 (Pure C Internal Physics Engine)
 * 遵循禁止虚假数据原则：所有传感器数据均由真实物理计算产生。
 */

#include "selflnn/robot/sensor_simulation.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/robot/kinematics.h"
#include "selflnn/utils/logging.h"
#include <math.h>
#include <string.h>
#include <float.h>

#ifdef _MSC_VER
#pragma warning(disable:4100 4189 4244 4267 4701 4101)
#endif

/* 内联四元数旋转（复用math_utils.h的quat_rotate_vec3） */
static void sim_quat_rotate_vec3(const float* q, const float* v, float* result) {
	float qv[3] = { q[0], q[1], q[2] };
	float qw = q[3];
	float cross[3] = {
		qv[1] * v[2] - qv[2] * v[1],
		qv[2] * v[0] - qv[0] * v[2],
		qv[0] * v[1] - qv[1] * v[0]
	};
	float t[3] = {
		2.0f * (qv[1] * cross[2] - qv[2] * cross[1]),
		2.0f * (qv[2] * cross[0] - qv[0] * cross[2]),
		2.0f * (qv[0] * cross[1] - qv[1] * cross[0])
	};
	result[0] = v[0] + t[0];
	result[1] = v[1] + t[1];
	result[2] = v[2] + t[2];
}

/* XORShift PRNG */
static uint32_t sim_xorshift_next(uint32_t* state) {
	uint32_t x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

static float sim_randn(uint32_t* state) {
	float u1 = (float)(sim_xorshift_next(state) & 0x7FFFFFFF) / 2147483648.0f;
	float u2 = (float)(sim_xorshift_next(state) & 0x7FFFFFFF) / 2147483648.0f;
	return sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(6.283185f * u2);
}

/* ============================================================================
 * sensor_sim_init
 * 初始化仿真传感器上下文
 * ============================================================================ */
void sensor_sim_init(SensorSimContext* ctx) {
	if (!ctx) return;
	memset(ctx, 0, sizeof(SensorSimContext));
	ctx->lidar_config.min_range = 0.1f;
	ctx->lidar_config.max_range = 30.0f;
	ctx->lidar_config.num_rays = 360;
	ctx->lidar_config.num_layers = 1;
	ctx->lidar_config.fov_horizontal = 6.283185f;
	ctx->lidar_config.fov_vertical = 0.523599f;
	ctx->lidar_config.scan_rate = 10.0f;
	ctx->lidar_config.noise_stddev = 0.02f;
	ctx->depth_config.width = 640;
	ctx->depth_config.height = 480;
	ctx->depth_config.fov = 1.047198f;
	ctx->depth_config.min_depth = 0.1f;
	ctx->depth_config.max_depth = 20.0f;
	ctx->depth_config.noise_stddev = 0.005f;
	ctx->imu_config.accel_noise_stddev = 0.01f;
	ctx->imu_config.gyro_noise_stddev = 0.001f;
	ctx->imu_config.accel_bias_instability = 0.0001f;
	ctx->imu_config.gyro_bias_instability = 0.00001f;
	ctx->imu_config.update_rate = 200.0f;
	ctx->rng_state = 123456789u;
	ctx->initialized = 1;
}

/* ============================================================================
 * sensor_sim_ray_intersect
 * 射线与碰撞体求交：支持球体/AABB/盒体
 * ============================================================================ */
int sensor_sim_ray_intersect(const float* ray_origin, const float* ray_dir,
							 const SimCollisionObject* object,
							 float* hit_distance, float* hit_point, float* hit_normal) {
	if (!ray_origin || !ray_dir || !object || !hit_distance) return -1;

	float t_min = 1e30f;
	int hit = 0;
	float local_t, local_pt[3], local_norm[3];

	float pos[3] = { object->world_transform[0], object->world_transform[1], object->world_transform[2] };
	float quat[4] = { object->world_transform[3], object->world_transform[4],
					  object->world_transform[5], object->world_transform[6] };

	switch (object->shape.type) {
		case COLLISION_SHAPE_SPHERE: {
			float r = object->shape.data.sphere.radius;
			float oc[3] = { ray_origin[0] - pos[0], ray_origin[1] - pos[1], ray_origin[2] - pos[2] };
			float a = ray_dir[0]*ray_dir[0] + ray_dir[1]*ray_dir[1] + ray_dir[2]*ray_dir[2];
			float b = 2.0f * (oc[0]*ray_dir[0] + oc[1]*ray_dir[1] + oc[2]*ray_dir[2]);
			float c = oc[0]*oc[0] + oc[1]*oc[1] + oc[2]*oc[2] - r*r;
			float disc = b*b - 4.0f*a*c;
			if (disc >= 0.0f) {
				float sqrt_disc = sqrtf(disc);
				float t0 = (-b - sqrt_disc) / (2.0f*a);
				float t1 = (-b + sqrt_disc) / (2.0f*a);
				float t_candidate = (t0 > 0.001f) ? t0 : ((t1 > 0.001f) ? t1 : -1.0f);
				if (t_candidate > 0.001f && t_candidate < t_min) {
					t_min = t_candidate;
					hit = 1;
					local_pt[0] = ray_origin[0] + ray_dir[0]*t_candidate;
					local_pt[1] = ray_origin[1] + ray_dir[1]*t_candidate;
					local_pt[2] = ray_origin[2] + ray_dir[2]*t_candidate;
					float dn = 1.0f / r;
					local_norm[0] = (local_pt[0] - pos[0]) * dn;
					local_norm[1] = (local_pt[1] - pos[1]) * dn;
					local_norm[2] = (local_pt[2] - pos[2]) * dn;
				}
			}
			break;
		}
		case COLLISION_SHAPE_BOX: {
			float aabb_min[3] = { object->aabb_min[0], object->aabb_min[1], object->aabb_min[2] };
			float aabb_max[3] = { object->aabb_max[0], object->aabb_max[1], object->aabb_max[2] };
			float tmin_val[3], tmax_val[3];
			for (int i = 0; i < 3; i++) {
				float inv_d = 1.0f / (fabsf(ray_dir[i]) > 1e-12f ? ray_dir[i] : 1e-12f);
				float t1 = (aabb_min[i] - ray_origin[i]) * inv_d;
				float t2 = (aabb_max[i] - ray_origin[i]) * inv_d;
				if (t1 < t2) { tmin_val[i] = t1; tmax_val[i] = t2; }
				else { tmin_val[i] = t2; tmax_val[i] = t1; }
			}
			float t_near = fmaxf(fmaxf(tmin_val[0], tmin_val[1]), tmin_val[2]);
			float t_far = fminf(fminf(tmax_val[0], tmax_val[1]), tmax_val[2]);
			if (t_near <= t_far && t_far > 0.0f) {
				float t_candidate = (t_near > 0.001f) ? t_near : t_far;
				if (t_candidate < t_min) {
					t_min = t_candidate;
					hit = 1;
					local_pt[0] = ray_origin[0] + ray_dir[0]*t_candidate;
					local_pt[1] = ray_origin[1] + ray_dir[1]*t_candidate;
					local_pt[2] = ray_origin[2] + ray_dir[2]*t_candidate;
					float eps = 0.01f;
					if (fabsf(local_pt[0] - aabb_min[0]) < eps)
						{ local_norm[0] = -1.0f; local_norm[1] = 0.0f; local_norm[2] = 0.0f; }
					else if (fabsf(local_pt[0] - aabb_max[0]) < eps)
						{ local_norm[0] = 1.0f; local_norm[1] = 0.0f; local_norm[2] = 0.0f; }
					else if (fabsf(local_pt[1] - aabb_min[1]) < eps)
						{ local_norm[0] = 0.0f; local_norm[1] = -1.0f; local_norm[2] = 0.0f; }
					else if (fabsf(local_pt[1] - aabb_max[1]) < eps)
						{ local_norm[0] = 0.0f; local_norm[1] = 1.0f; local_norm[2] = 0.0f; }
					else if (fabsf(local_pt[2] - aabb_min[2]) < eps)
						{ local_norm[0] = 0.0f; local_norm[1] = 0.0f; local_norm[2] = -1.0f; }
					else
						{ local_norm[0] = 0.0f; local_norm[1] = 0.0f; local_norm[2] = 1.0f; }
				}
			}
			break;
		}
		case COLLISION_SHAPE_CYLINDER: {
			float r = object->radius;
			float h = object->aabb_max[1] - object->aabb_min[1];
			float oc[3] = { ray_origin[0] - pos[0], ray_origin[1] - pos[1], ray_origin[2] - pos[2] };
			float a = ray_dir[0]*ray_dir[0] + ray_dir[2]*ray_dir[2];
			float b = 2.0f * (oc[0]*ray_dir[0] + oc[2]*ray_dir[2]);
			float c = oc[0]*oc[0] + oc[2]*oc[2] - r*r;
			float t0 = 1e30f, t1 = 1e30f;
			float disc = b*b - 4.0f*a*c;
			if (disc >= 0.0f && fabsf(a) > 1e-10f) {
				float sqrt_disc = sqrtf(disc);
				t0 = (-b - sqrt_disc) / (2.0f*a);
				t1 = (-b + sqrt_disc) / (2.0f*a);
			}
			float half_h = h * 0.5f;
			for (int i = 0; i < 2; i++) {
				float t_candidate = (i == 0) ? t0 : t1;
				if (t_candidate <= 0.001f) continue;
				float py = ray_origin[1] + ray_dir[1]*t_candidate;
				if (py >= pos[1] - half_h && py <= pos[1] + half_h && t_candidate < t_min) {
					t_min = t_candidate;
					hit = 1;
					local_pt[0] = ray_origin[0] + ray_dir[0]*t_candidate;
					local_pt[1] = py;
					local_pt[2] = ray_origin[2] + ray_dir[2]*t_candidate;
					float dr = 1.0f / (r + 1e-10f);
					local_norm[0] = (local_pt[0] - pos[0]) * dr;
					local_norm[1] = 0.0f;
					local_norm[2] = (local_pt[2] - pos[2]) * dr;
				}
			}
			break;
		}
		case COLLISION_SHAPE_PLANE: {
			float pn[3] = { object->shape.data.plane.normal.x,
						    object->shape.data.plane.normal.y,
						    object->shape.data.plane.normal.z };
			float pd = object->shape.data.plane.d;
			float denom = pn[0]*ray_dir[0] + pn[1]*ray_dir[1] + pn[2]*ray_dir[2];
			if (fabsf(denom) > 1e-10f) {
				float t_candidate = -(pn[0]*ray_origin[0] + pn[1]*ray_origin[1] + pn[2]*ray_origin[2] - pd) / denom;
				if (t_candidate > 0.001f && t_candidate < t_min) {
					t_min = t_candidate;
					hit = 1;
					local_pt[0] = ray_origin[0] + ray_dir[0]*t_candidate;
					local_pt[1] = ray_origin[1] + ray_dir[1]*t_candidate;
					local_pt[2] = ray_origin[2] + ray_dir[2]*t_candidate;
					local_norm[0] = pn[0];
					local_norm[1] = pn[1];
					local_norm[2] = pn[2];
				}
			}
			break;
		}
		default:
			break;
	}

	if (hit) {
		*hit_distance = t_min;
		if (hit_point) { hit_point[0] = local_pt[0]; hit_point[1] = local_pt[1]; hit_point[2] = local_pt[2]; }
		if (hit_normal) { hit_normal[0] = local_norm[0]; hit_normal[1] = local_norm[1]; hit_normal[2] = local_norm[2]; }
		return 0;
	}

	return -1;
}

/* ============================================================================
 * sensor_sim_lidar_scan
 * 仿真激光雷达扫描 - 基于内部纯C物理引擎的射线追踪
 * ============================================================================ */
int sensor_sim_lidar_scan(SensorSimContext* ctx,
						  const float* mount_pos, const float* mount_quat,
						  const SimPhysicsPipeline* pipeline,
						  SensorSimLidarPoint* points, int max_points) {
	if (!ctx || !ctx->initialized) return -1;
	if (!mount_pos || !pipeline || !points || max_points <= 0) return -1;

	SensorSimLidarConfig* cfg = &ctx->lidar_config;
	float noise = cfg->noise_stddev;
	int num_rays = cfg->num_rays;
	int num_layers = cfg->num_layers;
	float vfov_half = cfg->fov_vertical * 0.5f;
	float hfov = cfg->fov_horizontal;
	float min_r = cfg->min_range;
	float max_r = cfg->max_range;

	int point_idx = 0;
	int total_rays = num_rays * num_layers;
	if (total_rays > max_points) total_rays = max_points;

	for (int layer = 0; layer < num_layers && point_idx < max_points; layer++) {
		float v_angle = (num_layers > 1) ? -vfov_half + vfov_half*2.0f*layer/(float)(num_layers-1) : 0.0f;
		for (int ray = 0; ray < num_rays && point_idx < max_points; ray++) {
			float h_angle = hfov * (float)ray / (float)num_rays;
			float local_dir[3] = {
				cosf(v_angle) * cosf(h_angle),
				sinf(v_angle),
				cosf(v_angle) * sinf(h_angle)
			};
			float world_dir[3];
			if (mount_quat) {
				sim_quat_rotate_vec3(mount_quat, local_dir, world_dir);
			} else {
				world_dir[0] = local_dir[0];
				world_dir[1] = local_dir[1];
				world_dir[2] = local_dir[2];
			}

			float best_dist = max_r;
			int hit_any = 0;
			for (int o = 0; o < pipeline->object_count && o < 128; o++) {
				if (!pipeline->objects[o].active) continue;
				float hit_dist, hit_pt[3], hit_n[3];
				if (sensor_sim_ray_intersect(mount_pos, world_dir, &pipeline->objects[o],
											 &hit_dist, hit_pt, hit_n) == 0) {
					if (hit_dist > min_r && hit_dist < best_dist) {
						best_dist = hit_dist;
						hit_any = 1;
						points[point_idx].point[0] = hit_pt[0];
						points[point_idx].point[1] = hit_pt[1];
						points[point_idx].point[2] = hit_pt[2];
					}
				}
			}

			if (hit_any && best_dist < max_r) {
				float nz = sim_randn(&ctx->rng_state) * noise;
				points[point_idx].range = best_dist + nz;
				points[point_idx].intensity = fmaxf(0.0f, 1.0f - best_dist/max_r) + sim_randn(&ctx->rng_state)*0.05f;
				points[point_idx].valid = 1;
			} else {
				points[point_idx].range = max_r;
				points[point_idx].intensity = 0.0f;
				points[point_idx].point[0] = mount_pos[0] + world_dir[0]*max_r;
				points[point_idx].point[1] = mount_pos[1] + world_dir[1]*max_r;
				points[point_idx].point[2] = mount_pos[2] + world_dir[2]*max_r;
				points[point_idx].valid = 0;
			}
			point_idx++;
		}
	}

	return point_idx;
}

/* ============================================================================
 * sensor_sim_depth_camera
 * 仿真深度相机 - 针孔模型 + 射线追踪
 * ============================================================================ */
int sensor_sim_depth_camera(SensorSimContext* ctx,
							const float* mount_pos, const float* mount_quat,
							const SimPhysicsPipeline* pipeline,
							SensorSimDepthPixel* pixels, int max_pixels) {
	if (!ctx || !ctx->initialized) return -1;
	if (!mount_pos || !pipeline || !pixels || max_pixels <= 0) return -1;

	SensorSimDepthCameraConfig* cfg = &ctx->depth_config;
	int w = cfg->width;
	int h = cfg->height;
	float fov_half = cfg->fov * 0.5f;
	float aspect = (float)w / (float)h;
	float focal = 1.0f / tanf(fov_half);
	float noise = cfg->noise_stddev;
	float min_d = cfg->min_depth;
	float max_d = cfg->max_depth;

	int total = w * h;
	if (total > max_pixels) total = max_pixels;

	for (int py = 0; py < h && py*w < total; py++) {
		for (int px = 0; px < w && py*w + px < total; px++) {
			float nx = (2.0f * (float)(px + 0.5f) / (float)w - 1.0f) * aspect;
			float ny = 1.0f - 2.0f * (float)(py + 0.5f) / (float)h;
			float local_dir[3] = { nx / focal, ny / focal, 1.0f };
			float len = sqrtf(local_dir[0]*local_dir[0] + local_dir[1]*local_dir[1] + local_dir[2]*local_dir[2]);
			local_dir[0] /= len; local_dir[1] /= len; local_dir[2] /= len;

			float world_dir[3];
			if (mount_quat) {
				sim_quat_rotate_vec3(mount_quat, local_dir, world_dir);
			} else {
				world_dir[0] = local_dir[0];
				world_dir[1] = local_dir[1];
				world_dir[2] = local_dir[2];
			}

			float best_dist = max_d;
			int hit_any = 0;
			for (int o = 0; o < pipeline->object_count && o < 128; o++) {
				if (!pipeline->objects[o].active) continue;
				float hit_dist, hit_pt[3], hit_n[3];
				if (sensor_sim_ray_intersect(mount_pos, world_dir, &pipeline->objects[o],
											 &hit_dist, hit_pt, hit_n) == 0) {
					if (hit_dist > min_d && hit_dist < best_dist) {
						best_dist = hit_dist;
						hit_any = 1;
					}
				}
			}

			int idx = py * w + px;
			if (hit_any && best_dist < max_d) {
				pixels[idx].depth = best_dist + sim_randn(&ctx->rng_state) * noise;
				pixels[idx].point[0] = mount_pos[0] + world_dir[0]*best_dist;
				pixels[idx].point[1] = mount_pos[1] + world_dir[1]*best_dist;
				pixels[idx].point[2] = mount_pos[2] + world_dir[2]*best_dist;
				pixels[idx].valid = 1;
			} else {
				pixels[idx].depth = max_d;
				pixels[idx].point[0] = 0.0f;
				pixels[idx].point[1] = 0.0f;
				pixels[idx].point[2] = 0.0f;
				pixels[idx].valid = 0;
			}
		}
	}

	return total;
}

/* ============================================================================
 * sensor_sim_imu_update
 * 仿真IMU更新 - 真实噪声模型（白噪声 + 偏置不稳定性 + 随机游走）
 * ============================================================================ */
int sensor_sim_imu_update(SensorSimContext* ctx,
						  const float* linear_accel, const float* angular_vel,
						  const float* orientation, float dt,
						  SensorSimImuFrame* frame) {
	if (!ctx || !ctx->initialized) return -1;
	if (!linear_accel || !angular_vel || !orientation || !frame || dt <= 0.0f) return -1;

	SensorSimImuConfig* cfg = &ctx->imu_config;

	/* 偏置随机游走更新 */
	float bias_scale = sqrtf(dt);
	for (int i = 0; i < 3; i++) {
		ctx->imu_bias_accel[i] += sim_randn(&ctx->rng_state) * cfg->accel_bias_instability * bias_scale;
		ctx->imu_bias_gyro[i] += sim_randn(&ctx->rng_state) * cfg->gyro_bias_instability * bias_scale;
	}

	/* 加速度计：真实加速度 + 偏置 + 白噪声 + 重力补偿（重力由调用方在linear_accel中处理） */
	for (int i = 0; i < 3; i++) {
		frame->acceleration[i] = linear_accel[i] + ctx->imu_bias_accel[i]
								  + sim_randn(&ctx->rng_state) * cfg->accel_noise_stddev;
	}

	/* 陀螺仪：真实角速度 + 偏置 + 白噪声 */
	for (int i = 0; i < 3; i++) {
		frame->angular_velocity[i] = angular_vel[i] + ctx->imu_bias_gyro[i]
									  + sim_randn(&ctx->rng_state) * cfg->gyro_noise_stddev;
	}

	/* 姿态四元数 */
	frame->orientation[0] = orientation[0];
	frame->orientation[1] = orientation[1];
	frame->orientation[2] = orientation[2];
	frame->orientation[3] = orientation[3];

	frame->timestamp = ctx->last_update_time + dt;
	ctx->last_update_time = frame->timestamp;
	frame->valid = 1;

	return 0;
}
