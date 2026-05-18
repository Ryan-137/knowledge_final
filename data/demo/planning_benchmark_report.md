# Planning Benchmark Report

- Generated At: `2026-04-22T10:09:55`
- Total Cases: `6`
- Gap Coverage: `1.00`
- Learning Path Coverage: `1.00`
- Action Template Coverage: `1.00`
- Simulation Positive Rate: `1.00`
- Adopt Non-Regression Rate: `1.00`
- Focus Match Rate: `1.00`
- Pass Rate: `1.00`

## Thresholds

- Gap Coverage >= `1.00`
- Learning Path Coverage >= `0.85`
- Action Template Coverage >= `1.00`
- Simulation Positive Rate >= `0.85`
- Adopt Non-Regression Rate >= `1.00`
- Focus Match Rate >= `0.85`

Adopt Basis uses OR semantics: `score+rank`, `score_only`, `rank_only`, `regressed`.

## Cases

| Case | Status | Target Role | Focus Nodes | Selected Actions | Sim Delta | Adopt Delta | Adopt Basis | Rank Change | Failure Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backend_bundle_nl | PASS | 后端开发工程师 (role_backend_engineer) | cap_backend_engineering, cap_backend_engineering | backend_gateway_hardening_project, devops_release_pipeline_project | 0.2541 | 0.2541 | score+rank | 2 -> 2 | - |
| data_engineer_structured | PASS | 数据工程师 (role_data_engineer) | cap_data_engineering, cap_data_engineering | data_pipeline_project | 0.3070 | 0.3070 | score+rank | 2 -> 1 | - |
| frontend_structured | PASS | 前端工程师 (role_frontend_engineer) | cap_frontend_engineering, cap_frontend_engineering | frontend_component_library_portfolio | 0.0903 | 0.0903 | score_only | 2 -> 4 | - |
| qa_nl | PASS | 测试开发工程师 (role_test_development_engineer) | cap_qa_engineering, cap_qa_engineering | qa_automation_project | 0.1786 | 0.1786 | score+rank | 2 -> 1 | - |
| appsec_structured | PASS | 应用安全工程师 (role_appsec_engineer) | cap_appsec_engineer, cap_appsec_engineer | appsec_review_project | 0.2931 | 0.2931 | score+rank | 1 -> 1 | - |
| devops_nl | PASS | DevOps 工程师 (role_devops_engineer) | cap_devops_engineering, cap_devops_engineering | devops_release_pipeline_project | 0.0628 | 0.0628 | score_only | 3 -> 5 | - |
