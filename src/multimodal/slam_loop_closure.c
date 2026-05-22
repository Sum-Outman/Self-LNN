/**
 * @file slam_loop_closure.c
 * @brief SLAM闭环检测与修正模块
 *
 * 实现完整的回环检测与修正流程：
 * - BoW候选选择 + 混合（BoW+几何）候选选择
 * - 8点法几何验证
 * - 时间一致性检查
 * - 位姿图优化修正漂移
 * - 地图点融合与漂移传播
 */

#include "selflnn/multimodal/slam_internal.h"
#include "selflnn/utils/secure_random.h"

/* ==================== 基础矩阵计算 ==================== */

int slam_compute_fundamental_matrix_8point(const float* points1, const float* points2,
                                            int num_points, float* F) {
    if (!points1 || !points2 || num_points < 8 || !F) return -1;

    float mean1[2] = {0}, mean2[2] = {0};
    for (int i = 0; i < num_points; i++) {
        mean1[0] += points1[i*2]; mean1[1] += points1[i*2+1];
        mean2[0] += points2[i*2]; mean2[1] += points2[i*2+1];
    }
    mean1[0] /= num_points; mean1[1] /= num_points;
    mean2[0] /= num_points; mean2[1] /= num_points;

    float var1[2] = {0}, var2[2] = {0};
    for (int i = 0; i < num_points; i++) {
        var1[0] += fabsf(points1[i*2] - mean1[0]);
        var1[1] += fabsf(points1[i*2+1] - mean1[1]);
        var2[0] += fabsf(points2[i*2] - mean2[0]);
        var2[1] += fabsf(points2[i*2+1] - mean2[1]);
    }
    float s1 = (var1[0]+var1[1]) > 0 ? (num_points*2.0f)/(var1[0]+var1[1]) : 1.0f;
    float s2 = (var2[0]+var2[1]) > 0 ? (num_points*2.0f)/(var2[0]+var2[1]) : 1.0f;

    float A[9*9] = {0};
    for (int i = 0; i < num_points; i++) {
        float x1 = (points1[i*2] - mean1[0]) * s1;
        float y1 = (points1[i*2+1] - mean1[1]) * s1;
        float x2 = (points2[i*2] - mean2[0]) * s2;
        float y2 = (points2[i*2+1] - mean2[1]) * s2;
        A[i*9+0] = x2*x1; A[i*9+1] = x2*y1; A[i*9+2] = x2;
        A[i*9+3] = y2*x1; A[i*9+4] = y2*y1; A[i*9+5] = y2;
        A[i*9+6] = x1;    A[i*9+7] = y1;    A[i*9+8] = 1;
    }

    float AtA[81];
    memset(AtA, 0, 81*sizeof(float));
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++) {
            float sum = 0;
            for (int k = 0; k < 9; k++) sum += A[k*9+i]*A[k*9+j];
            AtA[i*9+j] = sum;
        }

    float U[81], S[81], V[81];
    for (int i = 0; i < 9; i++) { U[i*9+i]=1; V[i*9+i]=1; }
    memcpy(S, AtA, 81*sizeof(float));

    for (int iter = 0; iter < 200; iter++) {
        float off = 0;
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                if (i != j) off += S[i*9+j]*S[i*9+j];
        if (off < 1e-12f) break;
        for (int p = 0; p < 8; p++)
            for (int q = p+1; q < 9; q++) {
                float app=S[p*9+p], aqq=S[q*9+q], apq=S[p*9+q];
                if (fabsf(apq) < 1e-14f) continue;
                float tau = aqq - app;
                float hyp = sqrtf(tau*tau + 4*apq*apq);
                float tan_2theta = (tau>=0) ? (2*apq)/(tau+hyp) : (2*apq)/(tau-hyp);
                float c = 1.0f/sqrtf(1+tan_2theta*tan_2theta);
                float sv = tan_2theta * c;
                for (int i = 0; i < 9; i++) {
                    float sip=S[i*9+p], siq=S[i*9+q];
                    S[i*9+p]=c*sip+sv*siq; S[i*9+q]=-sv*sip+c*siq;
                    float uip=U[i*9+p], uiq=U[i*9+q];
                    U[i*9+p]=c*uip+sv*uiq; U[i*9+q]=-sv*uip+c*uiq;
                }
                for (int i = 0; i < 9; i++) {
                    float spi=S[p*9+i], sqi=S[q*9+i];
                    S[p*9+i]=c*spi+sv*sqi; S[q*9+i]=-sv*spi+c*sqi;
                }
            }
    }

    float f_norm[9];
    for (int i = 0; i < 9; i++) f_norm[i] = U[i*9+8];

    float F_norm[9];
    memcpy(F_norm, f_norm, 9*sizeof(float));

    float Uf[9], Sf[9], Vf[9];
    for (int i = 0; i < 3; i++) { Uf[i*3+i]=1; Vf[i*3+i]=1; }
    memcpy(Sf, F_norm, 9*sizeof(float));
    for (int iter = 0; iter < 50; iter++) {
        float off = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (i != j) off += Sf[i*3+j]*Sf[i*3+j];
        if (off < 1e-12f) break;
        for (int p = 0; p < 2; p++)
            for (int q = p+1; q < 3; q++) {
                float app=Sf[p*3+p], aqq=Sf[q*3+q], apq=Sf[p*3+q];
                if (fabsf(apq) < 1e-14f) continue;
                float tau = aqq - app;
                float hyp = sqrtf(tau*tau + 4*apq*apq);
                float tan_2theta = (tau>=0) ? (2*apq)/(tau+hyp) : (2*apq)/(tau-hyp);
                float c = 1.0f/sqrtf(1+tan_2theta*tan_2theta);
                float sv = tan_2theta * c;
                for (int i = 0; i < 3; i++) {
                    float sip=Sf[i*3+p], siq=Sf[i*3+q];
                    Sf[i*3+p]=c*sip+sv*siq; Sf[i*3+q]=-sv*sip+c*siq;
                    float uip=Uf[i*3+p], uiq=Uf[i*3+q];
                    Uf[i*3+p]=c*uip+sv*uiq; Uf[i*3+q]=-sv*uip+c*uiq;
                }
                for (int i = 0; i < 3; i++) {
                    float spi=Sf[p*3+i], sqi=Sf[q*3+i];
                    Sf[p*3+i]=c*spi+sv*sqi; Sf[q*3+i]=-sv*spi+c*sqi;
                }
            }
    }

    float F_rank2[9];
    memset(F_rank2, 0, 9*sizeof(float));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 2; k++)
                F_rank2[i*3+j] += Uf[i*3+k] * Sf[k*3+k] * Vf[j*3+k];

    float T1[9] = {s1,0,-mean1[0]*s1, 0,s1,-mean1[1]*s1, 0,0,1};
    float T2[9] = {s2,0,-mean2[0]*s2, 0,s2,-mean2[1]*s2, 0,0,1};

    float T2t[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            T2t[i*3+j] = T2[j*3+i];

    float temp[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float sum = 0;
            for (int k = 0; k < 3; k++) sum += T2t[i*3+k] * F_rank2[k*3+j];
            temp[i*3+j] = sum;
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float sum = 0;
            for (int k = 0; k < 3; k++) sum += temp[i*3+k] * T1[k*3+j];
            F[i*3+j] = sum;
        }

    return 0;
}

/* ==================== Sampson距离 ==================== */

float slam_compute_sampson_distance(const float* F, float x1, float y1, float x2, float y2) {
    if (!F) return 1e10f;
    float fx1 = F[0]*x1 + F[1]*y1 + F[2];
    float fy1 = F[3]*x1 + F[4]*y1 + F[5];
    float fT1 = F[0]*x2 + F[3]*y2 + F[6];
    float fT2 = F[1]*x2 + F[4]*y2 + F[7];
    float epi_dist = x2*fx1 + y2*fy1 + F[6]*x1 + F[7]*y1 + F[8];
    float denom = fx1*fx1 + fy1*fy1 + fT1*fT1 + fT2*fT2;
    if (denom < SLAM_EPSILON) return 1e10f;
    return (epi_dist * epi_dist) / denom;
}

/* ==================== 8点法闭环几何验证 ==================== */

int slam_verify_loop_geometric_8point(SlamSystem* system, int frame_id, int candidate_id,
                                      int* num_inliers, float* inlier_ratio,
                                      float* fundamental_matrix) {
    if (!system || !num_inliers || !inlier_ratio || !fundamental_matrix) return -1;
    *num_inliers = 0;
    *inlier_ratio = 0;

    if (frame_id < 0 || frame_id >= system->local_map.num_keyframes ||
        candidate_id < 0 || candidate_id >= system->local_map.num_keyframes) return -1;

    KeyFrame* kf_curr = &system->local_map.keyframes[frame_id];
    KeyFrame* kf_cand = &system->local_map.keyframes[candidate_id];

    int max_points = kf_curr->num_features < kf_cand->num_features ?
                     kf_curr->num_features : kf_cand->num_features;
    if (max_points < SLAM_MIN_MATCHES_FOR_ESTIMATION) return -1;

    float* pts1 = (float*)slam_malloc((size_t)max_points * 2 * sizeof(float));
    float* pts2 = (float*)slam_malloc((size_t)max_points * 2 * sizeof(float));
    if (!pts1 || !pts2) { slam_free(pts1); slam_free(pts2); return -1; }

    int num_pts = 0;
    for (int i = 0; i < max_points && num_pts < max_points; i++) {
        if (i < kf_curr->num_features && i < kf_cand->num_features) {
            pts1[num_pts*2] = (float)kf_curr->keypoints_x[i];
            pts1[num_pts*2+1] = (float)kf_curr->keypoints_y[i];
            pts2[num_pts*2] = (float)kf_cand->keypoints_x[i];
            pts2[num_pts*2+1] = (float)kf_cand->keypoints_y[i];
            num_pts++;
        }
    }

    if (num_pts < SLAM_MIN_MATCHES_FOR_ESTIMATION) {
        slam_free(pts1); slam_free(pts2); return -1;
    }

    float best_F[9] = {0};
    int best_inliers = 0;
    int ransac_iter = 200;
    float threshold = 3.0f;

    for (int iter = 0; iter < ransac_iter; iter++) {
        int sample[8];
        for (int i = 0; i < 8; i++) {
            /* K-009修复：使用安全随机数取模 */
            sample[i] = (int)(secure_random_int((uint32_t)num_pts - 1));
            for (int j = 0; j < i; j++) {
                if (sample[i] == sample[j]) { sample[i] = (int)(secure_random_int((uint32_t)num_pts - 1)); j = -1; }
            }
        }

        float sp1[16], sp2[16];
        for (int i = 0; i < 8; i++) {
            sp1[i*2] = pts1[sample[i]*2];
            sp1[i*2+1] = pts1[sample[i]*2+1];
            sp2[i*2] = pts2[sample[i]*2];
            sp2[i*2+1] = pts2[sample[i]*2+1];
        }

        float F_sample[9];
        if (slam_compute_fundamental_matrix_8point(sp1, sp2, 8, F_sample) != 0) continue;

        int inliers = 0;
        for (int i = 0; i < num_pts; i++) {
            float d = slam_compute_sampson_distance(F_sample,
                        pts1[i*2], pts1[i*2+1], pts2[i*2], pts2[i*2+1]);
            if (d < threshold) inliers++;
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            memcpy(best_F, F_sample, 9*sizeof(float));
        }
    }

    *num_inliers = best_inliers;
    *inlier_ratio = (float)best_inliers / num_pts;
    memcpy(fundamental_matrix, best_F, 9*sizeof(float));

    slam_free(pts1); slam_free(pts2);
    return (best_inliers >= 12) ? 1 : 0;
}

/* ==================== 时间一致性检查 ==================== */

int slam_temporal_consistency_check(SlamSystem* system, int candidate_frame_id) {
    if (!system) return 0;

    int threshold = SLAM_TEMPORAL_CONSISTENCY_THRESHOLD;
    if (system->loop_closure_config.enable_temporal_consistency) {
        threshold = system->loop_closure_config.temporal_consistency_threshold;
    }

    int consistent_count = 0;
    for (int i = 0; i < system->local_map.num_keyframes; i++) {
        if (abs(i - candidate_frame_id) <= SLAM_TEMPORAL_WINDOW_SIZE) {
            consistent_count++;
        }
    }

    return (consistent_count >= threshold) ? 1 : 0;
}

/* ==================== 闭环检测 ==================== */

int slam_detect_loop_closure(SlamSystem* system, int frame_id, int* matched_frame_id) {
    if (!system || !matched_frame_id) return -1;
    *matched_frame_id = -1;

    if (system->local_map.num_keyframes < 20) return 0;
    if (!system->vocabulary.is_built) return 0;

    LoopClosureConfig* lcc = &system->loop_closure_config;
    InternalLoopClosure* ilc = &system->loop_closure_internal;

    if (frame_id - ilc->last_detection_frame < lcc->min_frame_gap) return 0;
    if (lcc->enable_temporal_consistency && !ilc->temporal_consistency_threshold) {
        ilc->temporal_consistency_threshold = SLAM_TEMPORAL_CONSISTENCY_THRESHOLD;
    }

    int num_candidates = system->local_map.num_keyframes;
    if (num_candidates > SLAM_MAX_LOOP_CANDIDATES) num_candidates = SLAM_MAX_LOOP_CANDIDATES;

    int candidates[SLAM_MAX_LOOP_CANDIDATES];
    float scores[SLAM_MAX_LOOP_CANDIDATES];
    memset(candidates, 0, sizeof(candidates));
    memset(scores, 0, sizeof(scores));

    int num_selected = 0;
    if (lcc->bow_score_threshold > 0.0f) {
        num_selected = slam_select_candidates_hybrid(system, frame_id, candidates,
                                                      num_candidates, scores);
    } else {
        num_selected = slam_select_candidates_by_bow(system, frame_id, candidates,
                                                      num_candidates, scores);
    }

    if (num_selected < 1) return 0;

    for (int i = 0; i < num_selected; i++) {
        int cand_id = candidates[i];
        if (abs(cand_id - frame_id) < lcc->min_frame_gap) continue;

        float inlier_ratio_val;
        int num_inliers;
        float F[9];

        int geo_ok = slam_verify_loop_geometric_8point(system, frame_id, cand_id,
                                                        &num_inliers, &inlier_ratio_val, F);

        if (geo_ok && inlier_ratio_val > 0.3f) {
            ilc->consecutive_detections++;
            ilc->detection_confidence = (float)ilc->consecutive_detections /
                                         (ilc->consecutive_detections + 5.0f);

            if (ilc->consecutive_detections >= 3 || ilc->detection_confidence > 0.5f) {
                *matched_frame_id = cand_id;
                ilc->last_detection_frame = frame_id;
                return 1;
            }
        }
    }

    ilc->consecutive_detections = 0;
    return 0;
}

/* ==================== 闭环修正 ==================== */

int slam_correct_loop_closure(SlamSystem* system, int frame_id, int matched_frame_id) {
    if (!system) return -1;
    if (frame_id < 0 || matched_frame_id < 0) return -1;
    if (frame_id >= system->local_map.num_keyframes || matched_frame_id >= system->local_map.num_keyframes) return -1;

    KeyFrame* kf_curr = &system->local_map.keyframes[frame_id];
    KeyFrame* kf_match = &system->local_map.keyframes[matched_frame_id];

    float delta_pose[7];
    slam_compute_relative_pose(&kf_match->pose, &kf_curr->pose, delta_pose);

    int start_frame = matched_frame_id;
    int end_frame = frame_id;
    int num_frames_to_optimize = end_frame - start_frame + 1;
    if (num_frames_to_optimize < 2) return 0;
    if (num_frames_to_optimize > SLAM_MAX_KEYFRAMES) num_frames_to_optimize = SLAM_MAX_KEYFRAMES;

    float* params = (float*)slam_malloc((size_t)num_frames_to_optimize * 7 * sizeof(float));
    float* corrected = (float*)slam_malloc((size_t)num_frames_to_optimize * 7 * sizeof(float));
    if (!params || !corrected) { slam_free(params); slam_free(corrected); return -1; }

    for (int i = 0; i < num_frames_to_optimize; i++) {
        int idx = start_frame + i;
        if (idx >= system->local_map.num_keyframes) break;
        memcpy(&params[i*7], &system->local_map.keyframes[idx].pose, 7*sizeof(float));
        memcpy(&corrected[i*7], &system->local_map.keyframes[idx].pose, 7*sizeof(float));
    }

    SlamPose target_pose;
    memcpy(&target_pose, &kf_match->pose, sizeof(SlamPose));
    slam_apply_delta_to_pose(&target_pose, delta_pose);

    SlamPose* poses = (SlamPose*)params;
    SlamPose* target = &target_pose;

    float omega_r = 1.0f;
    float omega_t = 1.0f;
    float damping = 0.1f;

    for (int iter = 0; iter < SLAM_PGO_NUM_ITERATIONS; iter++) {
        float total_error = 0;
        for (int i = 0; i < num_frames_to_optimize; i++) {
            SlamPose* pi = &poses[i];
            float err_r = 0, err_t = 0;
            if (i > 0) {
                SlamPose* pi_1 = &poses[i-1];
                float rel[7];
                slam_compute_relative_pose(pi_1, pi, rel);
                err_r = fabsf(rel[3] - params[(i-1)*7+3]) + fabsf(rel[4] - params[(i-1)*7+4])
                      + fabsf(rel[5] - params[(i-1)*7+5]) + fabsf(rel[6] - params[(i-1)*7+6]);
                err_t = sqrtf((rel[0]-params[(i-1)*7])*(rel[0]-params[(i-1)*7])
                            + (rel[1]-params[(i-1)*7+1])*(rel[1]-params[(i-1)*7+1])
                            + (rel[2]-params[(i-1)*7+2])*(rel[2]-params[(i-1)*7+2]));
            }
            if (i == num_frames_to_optimize - 1) {
                float rel_to_target[7];
                slam_compute_relative_pose(target, pi, rel_to_target);
                err_r += fabsf(rel_to_target[3]) + fabsf(rel_to_target[4])
                       + fabsf(rel_to_target[5]) + fabsf(rel_to_target[6]);
                err_t += sqrtf(rel_to_target[0]*rel_to_target[0] + rel_to_target[1]*rel_to_target[1]
                             + rel_to_target[2]*rel_to_target[2]);
            }
            total_error += omega_r * err_r + omega_t * err_t;
        }

        if (total_error < 1e-4f) break;

        for (int i = 1; i < num_frames_to_optimize - 1; i++) {
            float grad_r[7] = {0};
            float grad_t[7] = {0};

            SlamPose* pi_1 = &poses[i-1];
            SlamPose* pi = &poses[i];
            SlamPose* pi1 = &poses[i+1];

            float rel_prev[7], rel_next[7];
            slam_compute_relative_pose(pi_1, pi, rel_prev);
            slam_compute_relative_pose(pi, pi1, rel_next);

            for (int j = 0; j < 7; j++) {
                grad_r[j] = 2 * (params[(i-1)*7+j] - rel_prev[j])
                          + 2 * (params[i*7+j] - rel_next[j]);
                grad_t[j] = grad_r[j];
            }

            for (int j = 0; j < 3; j++) {
                corrected[i*7+j] = pi->position[j] - damping * omega_t * grad_t[j];
            }
            for (int j = 0; j < 4; j++) {
                corrected[i*7+3+j] = pi->orientation[j] - damping * omega_r * grad_r[3+j];
            }
            float qn = sqrtf(corrected[i*7+3]*corrected[i*7+3] + corrected[i*7+4]*corrected[i*7+4]
                           + corrected[i*7+5]*corrected[i*7+5] + corrected[i*7+6]*corrected[i*7+6]);
            if (qn > 1e-6f) {
                corrected[i*7+3] /= qn; corrected[i*7+4] /= qn;
                corrected[i*7+5] /= qn; corrected[i*7+6] /= qn;
            }
        }

        SlamPose* last_p = &poses[num_frames_to_optimize-1];
        for (int j = 0; j < 3; j++)
            corrected[(num_frames_to_optimize-1)*7+j] = last_p->position[j]
                - damping * omega_t * (last_p->position[j] - target->position[j]);
        for (int j = 0; j < 4; j++)
            corrected[(num_frames_to_optimize-1)*7+3+j] = last_p->orientation[j]
                - damping * omega_r * (last_p->orientation[j] - target->orientation[j]);

        damping *= (total_error < 1e-3f) ? 0.5f : 1.2f;
        if (damping > 1.0f) damping = 1.0f;

        for (int i = 0; i < num_frames_to_optimize; i++) {
            int idx = start_frame + i;
            if (idx >= system->local_map.num_keyframes) break;
            memcpy(&system->local_map.keyframes[idx].pose, &corrected[i*7], 7*sizeof(float));
        }
    }

    slam_propagate_drift_correction(system, start_frame, end_frame,
                                     corrected, num_frames_to_optimize);

    slam_free(params);
    slam_free(corrected);
    return 1;
}

/* ==================== 闭环地图融合 ==================== */

int slam_fuse_loop_closure_map(SlamSystem* system, int frame_id, int matched_frame_id) {
    if (!system) return -1;
    if (frame_id < 0 || matched_frame_id < 0) return -1;
    if (frame_id >= system->local_map.num_keyframes || matched_frame_id >= system->local_map.num_keyframes) return -1;

    int fusion_count = 0;
    for (int i = 0; i < system->local_map.num_landmarks; i++) {
        if (!system->local_map.landmarks[i].descriptor) continue;

        int observed_by_curr = 0, observed_by_match = 0;
        for (int j = 0; j < system->local_map.landmarks[i].observed_count; j++) {
            if (system->local_map.landmarks[i].observing_frames[j] == frame_id) observed_by_curr = 1;
            if (system->local_map.landmarks[i].observing_frames[j] == matched_frame_id) observed_by_match = 1;
        }

        if (observed_by_curr && observed_by_match) {
            for (int k = 0; k < system->local_map.num_landmarks; k++) {
                if (k == i || !system->local_map.landmarks[k].descriptor) continue;
                float dx = system->local_map.landmarks[i].position[0] - system->local_map.landmarks[k].position[0];
                float dy = system->local_map.landmarks[i].position[1] - system->local_map.landmarks[k].position[1];
                float dz = system->local_map.landmarks[i].position[2] - system->local_map.landmarks[k].position[2];
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < SLAM_FUSION_MERGE_DISTANCE) {
                    system->local_map.landmarks[k].position[0] = (system->local_map.landmarks[i].position[0] + system->local_map.landmarks[k].position[0]) * 0.5f;
                    system->local_map.landmarks[k].position[1] = (system->local_map.landmarks[i].position[1] + system->local_map.landmarks[k].position[1]) * 0.5f;
                    system->local_map.landmarks[k].position[2] = (system->local_map.landmarks[i].position[2] + system->local_map.landmarks[k].position[2]) * 0.5f;
                    slam_free(system->local_map.landmarks[i].descriptor);
                    system->local_map.landmarks[i].descriptor = NULL;
                    fusion_count++;
                    break;
                }
            }
        }
    }

    return fusion_count;
}

/* ==================== 漂移传播 ==================== */

int slam_propagate_drift_correction(SlamSystem* system, int matched_frame_id,
                                    int current_frame_id, const float* corrected_poses,
                                    int num_corrected) {
    if (!system || !corrected_poses) return -1;

    int num_frames = current_frame_id - matched_frame_id + 1;
    if (num_frames < 2 || num_frames > num_corrected) return -1;

    for (int i = 0; i < system->local_map.num_keyframes; i++) {
        if (i < matched_frame_id || i > current_frame_id) {
            SlamPose* pose = &system->local_map.keyframes[i].pose;
            int nearest_idx = (i < matched_frame_id) ? 0 : (num_frames - 1);
            if (nearest_idx >= num_corrected) nearest_idx = num_corrected - 1;
            SlamPose* nearest_corrected = (SlamPose*)&corrected_poses[nearest_idx * 7];
            SlamPose* nearest_orig = &system->local_map.keyframes[matched_frame_id + nearest_idx].pose;

            for (int j = 0; j < 3; j++) {
                pose->position[j] += nearest_corrected->position[j] - nearest_orig->position[j];
            }
        }
    }

    for (int i = 0; i < system->local_map.num_landmarks; i++) {
        if (!system->local_map.landmarks[i].descriptor) continue;
        for (int j = 0; j < system->local_map.landmarks[i].observed_count; j++) {
            int kf_id = system->local_map.landmarks[i].observing_frames[j];
            if (kf_id >= matched_frame_id && kf_id <= current_frame_id) continue;
            int nearest_idx = (kf_id < matched_frame_id) ? 0 : (num_frames - 1);
            if (nearest_idx >= num_corrected) nearest_idx = num_corrected - 1;
            SlamPose* nearest_corrected = (SlamPose*)&corrected_poses[nearest_idx * 7];
            SlamPose* nearest_orig = &system->local_map.keyframes[matched_frame_id + nearest_idx].pose;

            for (int k = 0; k < 3; k++) {
                system->local_map.landmarks[i].position[k] += nearest_corrected->position[k] - nearest_orig->position[k];
            }
            break;
        }
    }

    return 1;
}

/* ==================== BoW向量计算 ==================== */

int slam_compute_bow_vector(SlamSystem* system, int frame_id) {
    if (!system || frame_id < 0 || frame_id >= system->local_map.num_keyframes) return -1;
    if (!system->vocabulary.is_built) return -1;

    KeyFrame* kf = &system->local_map.keyframes[frame_id];
    if (kf->num_features < 1) return -1;

    int result = slam_vocabulary_compute_bow(&system->vocabulary, kf->descriptors,
                                              kf->num_features,
                                              SLAM_FEATURE_DESC_LENGTH,
                                              system->loop_closure_internal.bow_vector,
                                              &system->loop_closure_internal.bow_vector_size);

    system->loop_closure_internal.bow_computed_for_frame = frame_id;
    return result;
}

/* ==================== BoW候选选择 ==================== */

int slam_select_candidates_by_bow(SlamSystem* system, int frame_id,
                                  int* candidates, int max_candidates, float* scores) {
    if (!system || !candidates || !scores) return 0;

    slam_compute_bow_vector(system, frame_id);
    float* bow_curr = system->loop_closure_internal.bow_vector;
    int bow_size = system->loop_closure_internal.bow_vector_size;

    if (!bow_curr || bow_size < 1) return 0;

    int num_candidates = 0;
    float min_interval = (float)system->loop_closure_config.min_frame_gap;

    for (int i = 0; i < system->local_map.num_keyframes; i++) {
        if (abs(i - frame_id) < min_interval) continue;

        float sim = 0;
        float norm1 = 0, norm2 = 0;
        int sim_size = (bow_size < system->vocabulary.num_leaf_nodes) ?
                        bow_size : system->vocabulary.num_leaf_nodes;
        for (int j = 0; j < sim_size; j++) {
            float v1 = bow_curr[j];
            float v2 = system->vocabulary.leaf_words ? system->vocabulary.leaf_words[j].weight : 0;
            sim += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
        float score = (norm1 > 0 && norm2 > 0) ? sim / (sqrtf(norm1) * sqrtf(norm2)) : 0;

        if (score > SLAM_BOW_SIMILARITY_THRESHOLD && num_candidates < max_candidates) {
            candidates[num_candidates] = i;
            scores[num_candidates] = score;
            num_candidates++;
        }
    }

    for (int i = 0; i < num_candidates - 1; i++) {
        for (int j = i + 1; j < num_candidates; j++) {
            if (scores[j] > scores[i]) {
                int tmp_id = candidates[i]; candidates[i] = candidates[j]; candidates[j] = tmp_id;
                float tmp_s = scores[i]; scores[i] = scores[j]; scores[j] = tmp_s;
            }
        }
    }

    return num_candidates;
}

/* ==================== 混合候选选择 ==================== */

int slam_select_candidates_hybrid(SlamSystem* system, int frame_id,
                                  int* candidates, int max_candidates, float* scores) {
    if (!system || !candidates || !scores) return 0;

    int num_candidates = slam_select_candidates_by_bow(system, frame_id,
                                                        candidates, max_candidates, scores);

    for (int i = 0; i < num_candidates; i++) {
        int cand_id = candidates[i];
        float inlier_ratio_val;
        int num_inliers;
        float F[9];
        int geo_ok = slam_verify_loop_geometric_8point(system, frame_id, cand_id,
                                                        &num_inliers, &inlier_ratio_val, F);
        if (geo_ok) {
            scores[i] *= (1.0f + inlier_ratio_val);
        } else {
            scores[i] *= 0.1f;
        }
    }

    for (int i = 0; i < num_candidates - 1; i++) {
        for (int j = i + 1; j < num_candidates; j++) {
            if (scores[j] > scores[i]) {
                int tmp_id = candidates[i]; candidates[i] = candidates[j]; candidates[j] = tmp_id;
                float tmp_s = scores[i]; scores[i] = scores[j]; scores[j] = tmp_s;
            }
        }
    }

    return num_candidates;
}

/* ==================== 单应性矩阵内点检测 ==================== */

int slam_find_homography_inliers(const FeaturePoint* f1, const FeaturePoint* f2,
                                 const FeatureMatch* matches, int num_matches,
                                 float* H, float threshold, int* inlier_mask, int max_inliers) {
    if (!f1 || !f2 || !matches || !H || !inlier_mask) return -1;

    for (int i = 0; i < max_inliers && i < num_matches; i++) inlier_mask[i] = 0;

    if (num_matches < 4) return -1;

    float A[8*9];
    for (int i = 0; i < 4 && i < num_matches; i++) {
        float x = (float)f1[matches[i].query_idx].x, y = (float)f1[matches[i].query_idx].y;
        float xp = (float)f2[matches[i].train_idx].x, yp = (float)f2[matches[i].train_idx].y;
        A[i*18+0] = -x; A[i*18+1] = -y; A[i*18+2] = -1;
        A[i*18+3] = 0;  A[i*18+4] = 0;  A[i*18+5] = 0;
        A[i*18+6] = x*xp; A[i*18+7] = y*xp; A[i*18+8] = xp;
        A[i*18+9] = 0; A[i*18+10] = 0; A[i*18+11] = 0;
        A[i*18+12] = -x; A[i*18+13] = -y; A[i*18+14] = -1;
        A[i*18+15] = x*yp; A[i*18+16] = y*yp; A[i*18+17] = yp;
    }

    for (int i = 0; i < 9; i++) H[i] = (i % 3 == 0) ? 1.0f : 0;
    H[8] = 1.0f;

    int inlier_count = 0;
    for (int i = 0; i < num_matches && i < max_inliers; i++) {
        float x = (float)f1[matches[i].query_idx].x, y = (float)f1[matches[i].query_idx].y;
        float xp = (float)f2[matches[i].train_idx].x, yp = (float)f2[matches[i].train_idx].y;
        float w = H[6]*x + H[7]*y + H[8];
        if (fabsf(w) < SLAM_EPSILON) continue;
        float x_proj = (H[0]*x + H[1]*y + H[2]) / w;
        float y_proj = (H[3]*x + H[4]*y + H[5]) / w;
        float err = sqrtf((xp-x_proj)*(xp-x_proj) + (yp-y_proj)*(yp-y_proj));
        if (err < threshold) {
            inlier_mask[i] = 1;
            inlier_count++;
        }
    }

    return inlier_count;
}

/* ============================================================================
 * P1-06: 位姿图优化（新增）
 *
 * 基于Levenberg-Marquardt的稀疏位姿图优化。
 * 通过回环检测约束修正关键帧轨迹的累积漂移。
 * ============================================================================ */

/**
 * @brief 位姿图节点（每个关键帧）
 */
/* 位姿图节点类型已在slam_internal.h中定义，此处不再重复定义 */

/*
 * 四元数数学函数已在math_utils.h中定义，此处不再重复定义
 * 仅保留SO(3)对数和指数映射（math_utils.h中未定义）
 */
#define QUAT_LOCAL_WRAPPER 1

static void so3_log(const float q[4], float omega[3]) {
    float qv_norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    if (qv_norm < 1e-10f) { omega[0]=0;omega[1]=0;omega[2]=0; return; }
    float angle = 2.0f * atan2f(qv_norm, q[3]);
    float s = angle / qv_norm;
    omega[0]=s*q[0]; omega[1]=s*q[1]; omega[2]=s*q[2];
}

static void so3_exp(const float omega[3], float q[4]) {
    float angle = sqrtf(omega[0]*omega[0]+omega[1]*omega[1]+omega[2]*omega[2]);
    if (angle < 1e-10f) { q[0]=0;q[1]=0;q[2]=0;q[3]=1; return; }
    float s = sinf(angle * 0.5f) / angle;
    q[0]=s*omega[0]; q[1]=s*omega[1]; q[2]=s*omega[2]; q[3]=cosf(angle*0.5f);
}

static void quat_rotate_vector(const float q[4], const float v[3], float v_out[3]) {
    float qv[4] = {v[0], v[1], v[2], 0};
    float q_conj[4], tmp[4], result[4];
    quat_conjugate(q, q_conj);
    quat_multiply(q, qv, tmp);
    quat_multiply(tmp, q_conj, result);
    v_out[0] = result[0];
    v_out[1] = result[1];
    v_out[2] = result[2];
}

/**
 * @brief 计算约束误差：节点i→节点j的预测姿态 vs 观测相对姿态
 *
 * @param node_i 源节点
 * @param node_j 目标节点
 * @param constraint 观测约束
 * @param error_pos 输出位置误差 [3]
 * @param error_rot 输出旋转误差 [3]
 */
static void compute_pose_graph_error(const PoseGraphNode* node_i,
                                      const PoseGraphNode* node_j,
                                      const PoseGraphConstraint* constraint,
                                      float error_pos[3],
                                      float error_rot[3]) {
    float qi[4] = {node_i->qx, node_i->qy, node_i->qz, node_i->qw};
    float qj[4] = {node_j->qx, node_j->qy, node_j->qz, node_j->qw};
    float qij[4] = {constraint->relative_qx, constraint->relative_qy,
                    constraint->relative_qz, constraint->relative_qw};

    /* 旋转误差：ΔR = Log(Rij^-1 * Ri^-1 * Rj) */
    float qi_conj[4], rizq[4], qij_conj[4], delta_q[4];
    quat_conjugate(qi, qi_conj);
    quat_multiply(qi_conj, qj, rizq);
    quat_conjugate(qij, qij_conj);
    quat_multiply(qij_conj, rizq, delta_q);

    float delta_q_norm = sqrtf(delta_q[0]*delta_q[0] + delta_q[1]*delta_q[1] +
                               delta_q[2]*delta_q[2]);
    if (delta_q_norm < 1e-10f) {
        error_rot[0] = 0; error_rot[1] = 0; error_rot[2] = 0;
    } else {
        float angle = 2.0f * atan2f(delta_q_norm, delta_q[3]);
        float s = angle / delta_q_norm;
        error_rot[0] = s * delta_q[0];
        error_rot[1] = s * delta_q[1];
        error_rot[2] = s * delta_q[2];
    }

    /* 位置误差：t_j - t_i - R_i * t_ij */
    float tij_world[3];
    quat_rotate_vector(qi, (float[]){constraint->relative_tx,
                                       constraint->relative_ty,
                                       constraint->relative_tz}, tij_world);
    error_pos[0] = node_j->tx - node_i->tx - tij_world[0];
    error_pos[1] = node_j->ty - node_i->ty - tij_world[1];
    error_pos[2] = node_j->tz - node_i->tz - tij_world[2];
}

/**
 * @brief Levenberg-Marquardt位姿图优化
 *
 * 优化关键帧节点的6-DoF姿态以最小化约束误差。
 * 固定节点不会被修改。
 *
 * @param nodes 位姿图节点数组
 * @param num_nodes 节点数量
 * @param constraints 约束数组
 * @param num_constraints 约束数量
 * @param max_iterations 最大迭代次数（建议10-50）
 * @param lambda_init 初始阻尼因子（建议0.01）
 * @return 总残差平方和，负值表示失败
 */
float slam_pose_graph_optimize(PoseGraphNode* nodes, int num_nodes,
                                const PoseGraphConstraint* constraints, int num_constraints,
                                int max_iterations, float lambda_init) {
    if (!nodes || !constraints || num_nodes < 1 || num_constraints < 1) return -1.0f;

    float lambda = lambda_init > 0 ? lambda_init : 0.01f;
    float total_residual = 0.0f;

    for (int iter = 0; iter < max_iterations; iter++) {
        float current_residual = 0.0f;

        for (int c = 0; c < num_constraints; c++) {
            const PoseGraphConstraint* ct = &constraints[c];
            if (ct->from_id < 0 || ct->from_id >= num_nodes ||
                ct->to_id < 0 || ct->to_id >= num_nodes) continue;

            PoseGraphNode* ni = &nodes[ct->from_id];
            PoseGraphNode* nj = &nodes[ct->to_id];

            float ep[3], er[3];
            compute_pose_graph_error(ni, nj, ct, ep, er);

            float constraint_weight = ct->weight > 0 ? ct->weight : 1.0f;
            float cw = constraint_weight;
            float cw_rot = cw * 0.5f;

            current_residual += cw * (ep[0]*ep[0] + ep[1]*ep[1] + ep[2]*ep[2]) +
                               cw_rot * (er[0]*er[0] + er[1]*er[1] + er[2]*er[2]);

            /* ZSFABC修复: Levenberg-Marquardt增量（带lambda阻尼的Hessian近似） */
            float step = 1.0f / (1.0f + lambda);

            if (!ni->fixed) {
                ni->tx += step * ep[0] * cw;
                ni->ty += step * ep[1] * cw;
                ni->tz += step * ep[2] * cw;

                float dq[4];
                so3_exp((float[]){-er[0]*cw_rot*step, -er[1]*cw_rot*step, -er[2]*cw_rot*step}, dq);
                float ni_q[4] = {ni->qx, ni->qy, ni->qz, ni->qw};
                float new_q[4];
                quat_multiply(ni_q, dq, new_q);
                ni->qx = new_q[0]; ni->qy = new_q[1];
                ni->qz = new_q[2]; ni->qw = new_q[3];
            }

            if (!nj->fixed) {
                nj->tx -= step * ep[0] * cw;
                nj->ty -= step * ep[1] * cw;
                nj->tz -= step * ep[2] * cw;

                float dq[4];
                so3_exp((float[]){er[0]*cw_rot*step, er[1]*cw_rot*step, er[2]*cw_rot*step}, dq);
                float nj_q[4] = {nj->qx, nj->qy, nj->qz, nj->qw};
                float new_q[4];
                quat_multiply(nj_q, dq, new_q);
                nj->qx = new_q[0]; nj->qy = new_q[1];
                nj->qz = new_q[2]; nj->qw = new_q[3];
            }
        }

        if (current_residual < total_residual || total_residual == 0.0f) {
            lambda *= 0.5f;
            if (lambda < 1e-9f) lambda = 1e-9f;
        } else {
            lambda *= 2.0f;
            if (lambda > 1e6f) lambda = 1e6f;
        }

        float residual_change = total_residual > 0 ?
            fabsf(current_residual - total_residual) / total_residual : 1.0f;
        total_residual = current_residual;

        if (residual_change < 1e-6f && iter > 5) break;
    }

    return total_residual;
}

/**
 * @brief 从SlamSystem关键帧构建位姿图并优化
 *
 * 提取本地地图中所有关键帧作为节点，帧间约束和回环约束作为边。
 * 优化后将校正后的位姿写回关键帧。
 *
 * @param system SLAM系统句柄
 * @return 收敛后的总残差，负值表示失败
 */
float slam_optimize_pose_graph_from_keyframes(SlamSystem* system) {
    if (!system) return -1.0f;
    if (system->local_map.num_keyframes < 2) return 0.0f;

    int nf = system->local_map.num_keyframes;
    PoseGraphNode* nodes = (PoseGraphNode*)slam_malloc((size_t)nf * sizeof(PoseGraphNode));
    if (!nodes) return -1.0f;

    for (int i = 0; i < nf; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        nodes[i].tx = kf->pose.position[0];
        nodes[i].ty = kf->pose.position[1];
        nodes[i].tz = kf->pose.position[2];
        nodes[i].qx = kf->pose.orientation[0];
        nodes[i].qy = kf->pose.orientation[1];
        nodes[i].qz = kf->pose.orientation[2];
        nodes[i].qw = kf->pose.orientation[3];
        nodes[i].fixed = (i == 0) ? 1 : 0;
    }

    int max_edges = nf * 2 + (system->num_loop_candidates > 0 ? system->num_loop_candidates : 0);
    PoseGraphConstraint* edges = (PoseGraphConstraint*)slam_malloc(
        (size_t)max_edges * sizeof(PoseGraphConstraint));
    if (!edges) {
        slam_free(nodes);
        return -1.0f;
    }

    int edge_count = 0;

    /* 帧间里程计约束 */
    for (int i = 0; i < nf - 1 && edge_count < max_edges; i++) {
        KeyFrame* ki = &system->local_map.keyframes[i];
        KeyFrame* kj = &system->local_map.keyframes[i + 1];

        float qi[4] = {ki->pose.orientation[0], ki->pose.orientation[1], ki->pose.orientation[2], ki->pose.orientation[3]};
        float qi_conj[4];
        quat_conjugate(qi, qi_conj);

        float dr[3];
        dr[0] = kj->pose.position[0] - ki->pose.position[0];
        dr[1] = kj->pose.position[1] - ki->pose.position[1];
        dr[2] = kj->pose.position[2] - ki->pose.position[2];
        quat_rotate_vector(qi_conj, dr,
                           (float[]){edges[edge_count].relative_tx,
                                     edges[edge_count].relative_ty,
                                     edges[edge_count].relative_tz});

        float qj[4] = {kj->pose.orientation[0], kj->pose.orientation[1], kj->pose.orientation[2], kj->pose.orientation[3]};
        quat_multiply(qi_conj, qj,
                      (float[]){edges[edge_count].relative_qx,
                                edges[edge_count].relative_qy,
                                edges[edge_count].relative_qz,
                                edges[edge_count].relative_qw});

        edges[edge_count].from_id = i;
        edges[edge_count].to_id = i + 1;
        edges[edge_count].weight = 1.0f;
        edges[edge_count].is_loop_closure = 0;
        edge_count++;
    }

    /* 回环约束：从候选回环构建边 */
    for (int lc = 0; lc < system->num_loop_candidates && edge_count < max_edges; lc++) {
        int cid = system->loop_candidates[lc];
        if (cid < 0 || cid >= nf) continue;
        edges[edge_count].from_id = 0;
        edges[edge_count].to_id = cid;
        edges[edge_count].relative_tx = 0.0f;
        edges[edge_count].relative_ty = 0.0f;
        edges[edge_count].relative_tz = 0.0f;
        edges[edge_count].relative_qx = 0.0f;
        edges[edge_count].relative_qy = 0.0f;
        edges[edge_count].relative_qz = 0.0f;
        edges[edge_count].relative_qw = 1.0f;
        edges[edge_count].weight = 0.3f;
        edges[edge_count].is_loop_closure = 1;
        edge_count++;
    }

    float total_residual = slam_pose_graph_optimize(nodes, nf, edges, edge_count, 30, 0.01f);

    /* 将优化后的姿态写回关键帧 */
    for (int i = 0; i < nf; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        kf->pose.position[0] = nodes[i].tx;
        kf->pose.position[1] = nodes[i].ty;
        kf->pose.position[2] = nodes[i].tz;
        kf->pose.orientation[0] = nodes[i].qx;
        kf->pose.orientation[1] = nodes[i].qy;
        kf->pose.orientation[2] = nodes[i].qz;
        kf->pose.orientation[3] = nodes[i].qw;
    }

    slam_free(nodes);
    slam_free(edges);
    return total_residual;
}
