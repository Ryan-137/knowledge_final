# Recommendation Benchmark Report

- Generated At: `2026-04-22T10:09:54`
- Total Cases: `13`
- Hit@3: `1.00`
- Hit@5: `1.00`
- Forbidden Role Violations: `0`
- Explanation Coverage: `1.00`
- Provenance Coverage: `1.00`
- Fallback Coverage: `1.00`
- Pass Rate: `1.00`

## Thresholds

- Hit@3 >= `0.75`
- Hit@5 >= `0.90`
- Forbidden Role Violations = `0`
- Explanation Coverage >= `0.80`
- Provenance Coverage >= `0.80`
- Fallback Coverage >= `1.00`

## Cases

| Case | Status | Matched Role | Top Recommendations | Provenance | Failure Reasons |
| --- | --- | --- | --- | --- | --- |
| backend_nl | PASS | 后端开发工程师 (role_backend_engineer) | 后端开发工程师 (role_backend_engineer), 数据工程师 (role_data_engineer), 接口平台工程师 (role_api_platform_engineer) | 2 / onet_online, roadmap_sh | - |
| data_engineering_structured | PASS | 数据工程师 (role_data_engineer) | 数据工程师 (role_data_engineer), 离线数据工程师 (role_batch_data_engineer) | 1 / onet_online | - |
| analytics_nl | PASS | 数据分析师 (role_data_analyst) | 数据分析师 (role_data_analyst), 分析工程师 (role_analytics_engineer), BI 工程师 (role_bi_engineer) | 2 / onet_online, roadmap_sh | - |
| ml_nl | PASS | 计算机视觉工程师 (role_computer_vision_engineer) | 计算机视觉工程师 (role_computer_vision_engineer), NLP 工程师 (role_nlp_engineer), 大模型应用工程师 (role_llm_application_engineer) | 2 / onet_online, roadmap_sh | - |
| qa_nl | PASS | 质量平台工程师 (role_qa_platform_engineer) | 质量平台工程师 (role_qa_platform_engineer), 测试开发工程师 (role_test_development_engineer) | 2 / onet_online, roadmap_sh | - |
| platform_nl | PASS | 后端开发工程师 (role_backend_engineer) | 后端开发工程师 (role_backend_engineer), Rust 服务工程师 (role_rust_backend_engineer), 接口平台工程师 (role_api_platform_engineer) | 2 / onet_online, roadmap_sh | - |
| security_structured | PASS | 应用安全工程师 (role_appsec_engineer) | 应用安全工程师 (role_appsec_engineer), 云安全工程师 (role_cloud_security_engineer), 渗透测试工程师 (role_penetration_tester) | 2 / onet_online, roadmap_sh | - |
| frontend_structured | PASS | React 前端工程师 (role_react_frontend_engineer) | React 前端工程师 (role_react_frontend_engineer), Vue 前端工程师 (role_vue_frontend_engineer), 前端工程师 (role_frontend_engineer) | 2 / onet_online, roadmap_sh | - |
| android_structured | PASS | Android 工程师 (role_android_engineer) | Android 工程师 (role_android_engineer), iOS 工程师 (role_ios_engineer) | 1 / roadmap_sh | - |
| no_english_bridge | PASS | dir_quality_assurance, dir_data_analytics | - | 0 | - |
| math_only_bridge | PASS | cap_ml_engineering, cap_recommendation_algorithm_engineer | - | 0 | - |
| python_sql_sparse | PASS | cap_data_analysis, cap_data_engineering | - | 0 | - |
| backend_preference_sparse | PASS | role_api_platform_engineer, role_microservice_engineer, dir_fullstack, dir_web_backend | - | 0 | - |
