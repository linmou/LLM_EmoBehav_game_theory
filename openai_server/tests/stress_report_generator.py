#!/usr/bin/env python3
"""
Stress Test Report Generator

Generates comprehensive reports from stress test results,
including performance analysis, hang detection, and recommendations.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics


class StressReportGenerator:
    """Generates comprehensive stress test reports"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Report templates
        self.html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Server Stress Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; min-width: 120px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
        .section {{ margin: 30px 0; }}
        .section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .test-result {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }}
        .test-result.failed {{ border-left-color: #e74c3c; }}
        .hang-event {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
        .recommendation {{ background-color: #d4edda; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #28a745; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .chart-container {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""
    
    def generate_report(self, test_results: List[Any], summary: Dict[str, Any]) -> str:
        """Generate comprehensive stress test report"""
        timestamp = datetime.now()
        
        # Generate different report formats
        html_report = self._generate_html_report(test_results, summary, timestamp)
        json_report = self._generate_json_report(test_results, summary, timestamp)
        text_report = self._generate_text_report(test_results, summary, timestamp)
        
        # Save reports
        html_file = self.output_dir / f"stress_test_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        json_file = self.output_dir / f"stress_test_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        text_file = self.output_dir / f"stress_test_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
        
        html_file.write_text(html_report)
        json_file.write_text(json_report)
        text_file.write_text(text_report)
        
        # Create latest symlinks
        latest_html = self.output_dir / "latest_report.html"
        latest_json = self.output_dir / "latest_report.json"
        latest_text = self.output_dir / "latest_report.txt"
        
        # Remove existing symlinks and create new ones
        for latest_file, target_file in [(latest_html, html_file), (latest_json, json_file), (latest_text, text_file)]:
            if latest_file.exists():
                latest_file.unlink()
            latest_file.symlink_to(target_file.name)
        
        return str(html_file)
    
    def _generate_html_report(self, test_results: List[Any], summary: Dict[str, Any], timestamp: datetime) -> str:
        """Generate HTML report"""
        content = f"""
        <div class="header">
            <h1>🔥 Server Stress Test Report</h1>
            <p>Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Test Duration: {summary.get('test_duration', 0):.1f} seconds</p>
        </div>
        
        {self._generate_summary_section(summary)}
        {self._generate_test_results_section(test_results)}
        {self._generate_performance_analysis_section(summary)}
        {self._generate_hang_analysis_section(summary)}
        {self._generate_recommendations_section(test_results, summary)}
        {self._generate_detailed_data_section(test_results, summary)}
        """
        
        return self.html_template.format(content=content)
    
    def _generate_summary_section(self, summary: Dict[str, Any]) -> str:
        """Generate summary section for HTML report"""
        test_summary = summary.get('test_summary', {})
        perf_metrics = summary.get('performance_metrics', {})
        
        success_rate = test_summary.get('success_rate', 0)
        success_class = 'success' if success_rate >= 0.8 else 'warning' if success_rate >= 0.6 else 'error'
        
        avg_response_time = perf_metrics.get('avg_response_time', 0)
        response_class = 'success' if avg_response_time <= 5 else 'warning' if avg_response_time <= 15 else 'error'
        
        return f"""
        <div class="section">
            <h2>📊 Test Summary</h2>
            <div class="summary">
                <div class="metric-box">
                    <div class="metric-value {success_class}">{success_rate:.1%}</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{test_summary.get('total_tests', 0)}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value success">{test_summary.get('successful_tests', 0)}</div>
                    <div class="metric-label">Successful</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value error">{test_summary.get('failed_tests', 0)}</div>
                    <div class="metric-label">Failed</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value {response_class}">{avg_response_time:.2f}s</div>
                    <div class="metric-label">Avg Response Time</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{len(summary.get('detected_issues', []))}</div>
                    <div class="metric-label">Issues Detected</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_test_results_section(self, test_results: List[Any]) -> str:
        """Generate test results section"""
        results_html = []
        
        for result in test_results:
            status_class = "test-result" if result.success else "test-result failed"
            status_icon = "✅" if result.success else "❌"
            
            duration = ""
            if result.end_time and result.start_time:
                duration = f" ({(result.end_time - result.start_time).total_seconds():.1f}s)"
            
            error_msg = ""
            if not result.success and result.error_message:
                error_msg = f"<br><strong>Error:</strong> {result.error_message}"
            
            # Extract key metrics from result
            metrics_html = ""
            if result.metrics:
                key_metrics = self._extract_key_metrics(result.metrics)
                if key_metrics:
                    metrics_html = f"<br><strong>Metrics:</strong> {key_metrics}"
            
            results_html.append(f"""
            <div class="{status_class}">
                <h3>{status_icon} {result.test_name.replace('_', ' ').title()}{duration}</h3>
                <p><strong>Status:</strong> {'PASSED' if result.success else 'FAILED'}</p>
                {error_msg}
                {metrics_html}
            </div>
            """)
        
        return f"""
        <div class="section">
            <h2>🧪 Test Results</h2>
            {''.join(results_html)}
        </div>
        """
    
    def _generate_performance_analysis_section(self, summary: Dict[str, Any]) -> str:
        """Generate performance analysis section"""
        perf_metrics = summary.get('performance_metrics', {})
        server_health = summary.get('server_health', {})
        
        return f"""
        <div class="section">
            <h2>⚡ Performance Analysis</h2>
            
            <div class="chart-container">
                <h3>Response Time Statistics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Assessment</th></tr>
                    <tr>
                        <td>Average Response Time</td>
                        <td>{perf_metrics.get('avg_response_time', 0):.2f}s</td>
                        <td>{'🟢 Good' if perf_metrics.get('avg_response_time', 0) <= 5 else '🟡 Acceptable' if perf_metrics.get('avg_response_time', 0) <= 15 else '🔴 Poor'}</td>
                    </tr>
                    <tr>
                        <td>Maximum Response Time</td>
                        <td>{perf_metrics.get('max_response_time', 0):.2f}s</td>
                        <td>{'🟢 Good' if perf_metrics.get('max_response_time', 0) <= 30 else '🟡 Acceptable' if perf_metrics.get('max_response_time', 0) <= 60 else '🔴 Poor'}</td>
                    </tr>
                    <tr>
                        <td>Response Time Std Dev</td>
                        <td>{perf_metrics.get('response_time_std', 0):.2f}s</td>
                        <td>{'🟢 Consistent' if perf_metrics.get('response_time_std', 0) <= 5 else '🟡 Variable' if perf_metrics.get('response_time_std', 0) <= 15 else '🔴 Inconsistent'}</td>
                    </tr>
                </table>
            </div>
            
            <div class="chart-container">
                <h3>Resource Utilization</h3>
                <table>
                    <tr><th>Resource</th><th>Usage</th><th>Status</th></tr>
                    <tr>
                        <td>CPU Usage</td>
                        <td>{server_health.get('cpu_usage', 0):.1f}%</td>
                        <td>{'🟢 Normal' if server_health.get('cpu_usage', 0) <= 80 else '🟡 High' if server_health.get('cpu_usage', 0) <= 95 else '🔴 Critical'}</td>
                    </tr>
                    <tr>
                        <td>Memory Usage</td>
                        <td>{server_health.get('memory_usage', 0):.1f}%</td>
                        <td>{'🟢 Normal' if server_health.get('memory_usage', 0) <= 80 else '🟡 High' if server_health.get('memory_usage', 0) <= 95 else '🔴 Critical'}</td>
                    </tr>
                    <tr>
                        <td>Network Connections</td>
                        <td>{server_health.get('network_connections', 0)}</td>
                        <td>{'🟢 Normal' if server_health.get('network_connections', 0) <= 50 else '🟡 High' if server_health.get('network_connections', 0) <= 100 else '🔴 Critical'}</td>
                    </tr>
                </table>
            </div>
        </div>
        """
    
    def _generate_hang_analysis_section(self, summary: Dict[str, Any]) -> str:
        """Generate hang analysis section"""
        detected_issues = summary.get('detected_issues', [])
        
        if not detected_issues:
            return f"""
            <div class="section">
                <h2>🛡️ Hang Analysis</h2>
                <div class="test-result">
                    <p>✅ No hang conditions detected during testing</p>
                </div>
            </div>
            """
        
        hang_events = []
        for issue in detected_issues:
            hang_events.append(f"""
            <div class="hang-event">
                <h4>⚠️ {issue.get('hang_type', 'Unknown')} Detected</h4>
                <p><strong>Duration:</strong> {issue.get('timeout_duration', 0):.1f}s</p>
                <p><strong>Context:</strong> {issue.get('context', {})}</p>
                <p><strong>Resolved:</strong> {'Yes' if issue.get('resolved', False) else 'No'}</p>
            </div>
            """)
        
        return f"""
        <div class="section">
            <h2>🚨 Hang Analysis</h2>
            <p>Detected {len(detected_issues)} potential hang conditions:</p>
            {''.join(hang_events)}
        </div>
        """
    
    def _generate_recommendations_section(self, test_results: List[Any], summary: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        recommendations = self._analyze_and_recommend(test_results, summary)
        
        rec_html = []
        for rec in recommendations:
            rec_html.append(f"""
            <div class="recommendation">
                <h4>{rec['category']}</h4>
                <p><strong>Issue:</strong> {rec['issue']}</p>
                <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                <p><strong>Priority:</strong> {rec['priority']}</p>
            </div>
            """)
        
        return f"""
        <div class="section">
            <h2>💡 Recommendations</h2>
            {f"{''.join(rec_html)}" if rec_html else "<p>✅ No specific recommendations. Server performance appears optimal for the tested load patterns.</p>"}
        </div>
        """
    
    def _generate_detailed_data_section(self, test_results: List[Any], summary: Dict[str, Any]) -> str:
        """Generate detailed data section"""
        return f"""
        <div class="section">
            <h2>📋 Detailed Data</h2>
            
            <h3>Error Analysis</h3>
            <pre>{json.dumps(summary.get('error_analysis', {}), indent=2)}</pre>
            
            <h3>Server Health Snapshot</h3>
            <pre>{json.dumps(summary.get('server_health', {}), indent=2)}</pre>
            
            <h3>Raw Test Data</h3>
            <details>
                <summary>Click to view raw test results</summary>
                <pre>{json.dumps([self._serialize_result(r) for r in test_results], indent=2)}</pre>
            </details>
        </div>
        """
    
    def _generate_json_report(self, test_results: List[Any], summary: Dict[str, Any], timestamp: datetime) -> str:
        """Generate JSON report"""
        report_data = {
            "report_metadata": {
                "generated_at": timestamp.isoformat(),
                "report_version": "1.0",
                "test_duration": summary.get('test_duration', 0)
            },
            "summary": summary,
            "test_results": [self._serialize_result(r) for r in test_results],
            "recommendations": self._analyze_and_recommend(test_results, summary)
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_text_report(self, test_results: List[Any], summary: Dict[str, Any], timestamp: datetime) -> str:
        """Generate text report"""
        test_summary = summary.get('test_summary', {})
        perf_metrics = summary.get('performance_metrics', {})
        
        lines = [
            "=" * 60,
            "🔥 SERVER STRESS TEST REPORT",
            "=" * 60,
            f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Test Duration: {summary.get('test_duration', 0):.1f} seconds",
            "",
            "📊 SUMMARY",
            "-" * 30,
            f"Total Tests: {test_summary.get('total_tests', 0)}",
            f"Successful: {test_summary.get('successful_tests', 0)}",
            f"Failed: {test_summary.get('failed_tests', 0)}",
            f"Success Rate: {test_summary.get('success_rate', 0):.1%}",
            "",
            "⚡ PERFORMANCE METRICS",
            "-" * 30,
            f"Average Response Time: {perf_metrics.get('avg_response_time', 0):.2f}s",
            f"Maximum Response Time: {perf_metrics.get('max_response_time', 0):.2f}s",
            f"Response Time Std Dev: {perf_metrics.get('response_time_std', 0):.2f}s",
            "",
            "🧪 TEST RESULTS",
            "-" * 30
        ]
        
        for result in test_results:
            status = "PASS" if result.success else "FAIL"
            duration = ""
            if result.end_time and result.start_time:
                duration = f" ({(result.end_time - result.start_time).total_seconds():.1f}s)"
            
            lines.append(f"{'✅' if result.success else '❌'} {result.test_name}: {status}{duration}")
            
            if not result.success and result.error_message:
                lines.append(f"   Error: {result.error_message}")
        
        # Add hang analysis
        detected_issues = summary.get('detected_issues', [])
        if detected_issues:
            lines.extend([
                "",
                "🚨 HANG ANALYSIS",
                "-" * 30
            ])
            for issue in detected_issues:
                lines.append(f"⚠️  {issue.get('hang_type', 'Unknown')}: {issue.get('timeout_duration', 0):.1f}s")
        
        # Add recommendations
        recommendations = self._analyze_and_recommend(test_results, summary)
        if recommendations:
            lines.extend([
                "",
                "💡 RECOMMENDATIONS",
                "-" * 30
            ])
            for rec in recommendations:
                lines.append(f"• {rec['category']}: {rec['recommendation']} (Priority: {rec['priority']})")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _extract_key_metrics(self, metrics: Dict[str, Any]) -> str:
        """Extract key metrics for display"""
        key_metrics = []
        
        if isinstance(metrics, dict):
            # Look for common metric patterns
            if 'success_rate' in metrics:
                key_metrics.append(f"Success Rate: {metrics['success_rate']:.1%}")
            
            if 'avg_response_time' in metrics:
                key_metrics.append(f"Avg Response: {metrics['avg_response_time']:.2f}s")
            
            if 'max_successful_length' in metrics:
                key_metrics.append(f"Max Length: {metrics['max_successful_length']} chars")
            
            if 'concurrent_requests' in metrics:
                key_metrics.append(f"Concurrent: {metrics['concurrent_requests']}")
            
            # Check for nested results
            for key, value in metrics.items():
                if isinstance(value, dict) and 'success_rate' in value:
                    key_metrics.append(f"{key}: {value['success_rate']:.1%}")
        
        return ", ".join(key_metrics[:3])  # Limit to top 3 metrics
    
    def _analyze_and_recommend(self, test_results: List[Any], summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """Analyze results and generate recommendations"""
        recommendations = []
        
        test_summary = summary.get('test_summary', {})
        perf_metrics = summary.get('performance_metrics', {})
        server_health = summary.get('server_health', {})
        detected_issues = summary.get('detected_issues', [])
        
        # Success rate analysis
        success_rate = test_summary.get('success_rate', 0)
        if success_rate < 0.8:
            recommendations.append({
                "category": "Reliability",
                "issue": f"Low success rate ({success_rate:.1%})",
                "recommendation": "Investigate server configuration, resource allocation, or request timeout settings",
                "priority": "High" if success_rate < 0.6 else "Medium"
            })
        
        # Response time analysis
        avg_response_time = perf_metrics.get('avg_response_time', 0)
        if avg_response_time > 15:
            recommendations.append({
                "category": "Performance",
                "issue": f"Slow average response time ({avg_response_time:.2f}s)",
                "recommendation": "Consider optimizing model configuration, increasing GPU memory, or reducing batch size",
                "priority": "High" if avg_response_time > 30 else "Medium"
            })
        
        # Response time consistency
        response_time_std = perf_metrics.get('response_time_std', 0)
        if response_time_std > 10:
            recommendations.append({
                "category": "Consistency",
                "issue": f"High response time variability (std: {response_time_std:.2f}s)",
                "recommendation": "Review server load balancing, queue management, or request processing patterns",
                "priority": "Medium"
            })
        
        # Resource utilization
        cpu_usage = server_health.get('cpu_usage', 0)
        if cpu_usage > 90:
            recommendations.append({
                "category": "Resources",
                "issue": f"High CPU utilization ({cpu_usage:.1f}%)",
                "recommendation": "Consider scaling to multiple instances or optimizing CPU-intensive operations",
                "priority": "High"
            })
        
        memory_usage = server_health.get('memory_usage', 0)
        if memory_usage > 90:
            recommendations.append({
                "category": "Resources", 
                "issue": f"High memory utilization ({memory_usage:.1f}%)",
                "recommendation": "Monitor for memory leaks or consider increasing available memory",
                "priority": "High"
            })
        
        # GPU memory analysis
        gpu_memory_usage = server_health.get('gpu_memory_usage', {})
        if gpu_memory_usage:
            max_gpu_usage = max(gpu_memory_usage.values()) if gpu_memory_usage.values() else 0
            if max_gpu_usage > 95:
                recommendations.append({
                    "category": "GPU Resources",
                    "issue": f"Very high GPU memory usage ({max_gpu_usage:.1f}%)",
                    "recommendation": "Reduce gpu_memory_utilization setting or use smaller model for better headroom",
                    "priority": "High"
                })
        
        # Hang analysis
        if detected_issues:
            hang_types = {}
            for issue in detected_issues:
                hang_type = issue.get('hang_type', 'unknown')
                hang_types[hang_type] = hang_types.get(hang_type, 0) + 1
            
            for hang_type, count in hang_types.items():
                recommendations.append({
                    "category": "Stability",
                    "issue": f"Detected {count} {hang_type} events",
                    "recommendation": self._get_hang_recommendation(hang_type),
                    "priority": "High"
                })
        
        # Concurrent request analysis
        for result in test_results:
            if result.test_name == "basic_concurrent_load" and not result.success:
                recommendations.append({
                    "category": "Concurrency",
                    "issue": "Concurrent request handling issues",
                    "recommendation": "Adjust max_num_seqs, batch_size, or implement request queuing",
                    "priority": "Medium"
                })
        
        return recommendations
    
    def _get_hang_recommendation(self, hang_type: str) -> str:
        """Get specific recommendation for hang type"""
        recommendations = {
            "request_timeout": "Increase request timeout settings or optimize request processing",
            "server_unresponsive": "Check server health monitoring and implement automatic restart mechanisms",
            "queue_deadlock": "Review request queue management and concurrent processing limits",
            "resource_exhaustion": "Monitor and limit resource usage, implement circuit breakers",
            "streaming_hang": "Implement streaming response timeouts and chunk monitoring"
        }
        
        return recommendations.get(hang_type, "Review server logs and implement appropriate monitoring")
    
    def _serialize_result(self, result) -> Dict[str, Any]:
        """Serialize test result for JSON output"""
        return {
            "test_name": result.test_name,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "success": result.success,
            "error_message": result.error_message,
            "metrics": result.metrics,
            "server_state": result.server_state
        }


def main():
    """Test the report generator"""
    import argparse
    from datetime import datetime, timedelta
    
    # Create sample test results for demonstration
    class MockResult:
        def __init__(self, name, success, error=None, metrics=None):
            self.test_name = name
            self.start_time = datetime.now() - timedelta(minutes=10)
            self.end_time = datetime.now() - timedelta(minutes=5)
            self.success = success
            self.error_message = error
            self.metrics = metrics or {}
            self.server_state = {"cpu_usage": 45.2, "memory_usage": 67.8}
    
    sample_results = [
        MockResult("server_health_baseline", True, metrics={"response_time": 1.2, "success_rate": 1.0}),
        MockResult("basic_concurrent_load", True, metrics={"concurrent_10": {"success_rate": 0.9, "avg_response_time": 3.4}}),
        MockResult("progressive_context_length", False, "Context limit exceeded", {"max_successful_length": 15000}),
        MockResult("hang_detection_scenarios", True, metrics={"hang_scenarios_tested": 4, "detected_hangs": 0})
    ]
    
    sample_summary = {
        "test_summary": {"total_tests": 4, "successful_tests": 3, "failed_tests": 1, "success_rate": 0.75},
        "performance_metrics": {"avg_response_time": 8.5, "max_response_time": 25.3, "response_time_std": 12.1},
        "server_health": {"cpu_usage": 45.2, "memory_usage": 67.8, "gpu_memory_usage": {0: 85.3}},
        "detected_issues": [],
        "test_duration": 300
    }
    
    generator = StressReportGenerator("sample_reports")
    report_file = generator.generate_report(sample_results, sample_summary)
    
    print(f"Sample report generated: {report_file}")
    print("Open the HTML file in a browser to view the full report.")


if __name__ == "__main__":
    main()