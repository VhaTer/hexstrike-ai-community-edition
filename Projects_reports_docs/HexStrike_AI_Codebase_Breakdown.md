# HexStrike AI Community Edition - Codebase Breakdown

## Overview

This document provides a comprehensive per-file breakdown of the HexStrike AI Community Edition codebase, detailing each file's role, key imports, public classes/functions, and relationships to other modules. This breakdown is designed to give Python developers a clear mental map of the core infrastructure.

## Core Entry Points

### hexstrike_mcp.py

- **Role**: Main entry point for the MCP (Model Context Protocol) client
- **Key Imports**:
  - `shared.colored_formatter` for logging formatting
  - `mcp_core.mcp_entry` for MCP server setup
  - `mcp_core.args` for command line argument parsing
- **Public Functions**:
  - `main()`: Entry point that parses arguments and runs the MCP server
- **Relationships**:
  - Acts as the primary interface for MCP communication
  - Initializes logging setup with colored formatting
  - Calls `run_mcp()` from mcp_core module

### hexstrike_server.py

- **Role**: Standalone FastMCP 3.x server with native HTTP/SSE transport
- **Key Imports**:
  - `starlette.responses` for HTTP responses
  - `starlette.staticfiles` for serving static files
  - `mcp_core.server_setup` for server initialization
  - `server_core.*` modules for core functionality
- **Public Functions**:
  - `_build_dashboard_response()`: Builds web dashboard data
  - `_json_status_response()`: Returns JSON with HTTP status
  - `register_http_routes()`: Registers HTTP routes for dashboard and health checks
- **Relationships**:
  - Main server entry point that initializes FastMCP server
  - Serves web dashboard via Starlette
  - Integrates with core server components for telemetry and monitoring
  - Provides health check endpoints

### config.py

- **Role**: Global configuration management
- **Key Imports**: None
- **Public Data Structures**:
  - `_config`: Dictionary containing all configuration values
- **Configuration Values**:
  - App name and version
  - Data directory paths
  - Timeout settings
  - Cache configuration
  - Wordlist definitions for password cracking and directory enumeration
- **Relationships**:
  - Provides centralized configuration for all modules
  - Defines paths for security tools and wordlists
  - Used by other modules for initialization and settings

### tool_registry.py

- **Role**: Tool registry with compact schemas for HexStrike AI Agent
- **Key Imports**:
  - `typing` for type hints
  - `logging` for logging
- **Public Data Structures**:
  - `TOOLS`: Dictionary mapping tool names to their definitions
- **Tool Categories**:
  - Vulnerability Intelligence (vulnx)
  - Intelligence (analyze-target, create-attack-chain, smart-scan, technology-detection)
  - Network Recon (nmap, masscan, rustscan, etc.)
  - Web Recon (gobuster, ffuf, feroxbuster, etc.)
  - Web Vulnerability (nuclei, nikto, sqlmap, etc.)
  - Exploitation (commix, msfvenom, metasploit)
  - Brute Force (hydra, hashcat, john, etc.)
  - Active Directory (ldapdomaindump, certipy, bloodhound, etc.)
  - OSINT (whois, amass, subfinder, etc.)
  - WiFi Pentest (aircrack-ng, airodump-ng, etc.)
  - Binary Analysis (checksec, binwalk, ropgadget, etc.)
  - Cloud (pacu, prowler, trivy, etc.)
- **Relationships**:
  - Central registry for all security tools
  - Used by MCP tools to execute commands
  - Defines effectiveness ratings and parameters for each tool

## MCP Core Modules

### mcp_core/mcp_entry.py

- **Role**: Entry point for MCP server setup
- **Key Imports**:
  - `mcp_core.server_setup` for server setup
- **Public Functions**:
  - `run_mcp(args, logger)`: Runs the HexStrike MCP server in standalone mode
- **Relationships**:
  - Called by hexstrike_mcp.py
  - Initializes the standalone MCP server
  - Handles stdio transport for Claude Desktop

### mcp_core/server_setup.py

- **Role**: MCP server setup and initialization
- **Key Imports**: Various MCP core modules
- **Public Functions**:
  - `setup_mcp_server_standalone(logger)`: Sets up MCP server in standalone mode
- **Relationships**:
  - Used by mcp_entry.py
  - Initializes various MCP tool integrations
  - Configures server capabilities

### mcp_core/ Direct Tool Modules

Each file in mcp_core implements direct tool integrations:

- **active_directory_direct.py**: Active Directory enumeration tools
- **exploit_framework_direct.py**: Exploitation framework tools
- **net_scan_direct.py**: Network scanning tools
- **osint_direct.py**: OSINT tools
- **password_cracking_direct.py**: Password cracking tools
- **recon_direct.py**: Reconnaissance tools
- **security_direct.py**: Security scanning tools
- **smb_enum_direct.py**: SMB enumeration tools
- **technology_detector_direct.py**: Technology detection tools
- **testssl_direct.py**: SSL/TLS testing tools
- **web_fuzz_direct.py**: Web fuzzing tools
- **web_probe_direct.py**: Web probing tools
- **web_recon_direct.py**: Web reconnaissance tools
- **web_scan_direct.py**: Web scanning tools
- **wifi_direct.py**: WiFi testing tools

Each follows a similar pattern:

- **Role**: Direct integration with specific security tools
- **Key Imports**: Tool-specific libraries
- **Public Functions**: Tool-specific execution functions
- **Relationships**: Called by MCP server to execute tools

## Server Core Modules

### server_core/singletons.py

- **Role**: Central singleton instances for shared services
- **Key Imports**: All server core modules
- **Public Instances**:
  - `cache`: HexStrikeCache instance
  - `session_store`: SessionStore instance
  - `wordlist_store`: WordlistStore instance
  - `run_history`: RunHistoryStore instance
  - `tool_stats`: ToolStatsStore instance
  - `enhanced_process_manager`: EnhancedProcessManager instance
  - `error_handler`: IntelligentErrorHandler instance
  - `degradation_manager`: GracefulDegradation instance
  - `cve_intelligence`: CVEIntelligenceManager instance
  - `exploit_generator`: AIExploitGenerator instance
  - `vulnerability_correlator`: VulnerabilityCorrelator instance
  - `decision_engine`: IntelligentDecisionEngine instance
  - `bugbounty_manager`: BugBountyWorkflowManager instance
  - `ctf_manager`: CTFWorkflowManager instance
- **Relationships**:
  - Central point for accessing shared services
  - Used by all server components
  - Ensures single instances across the application

### server_core/enhanced_command_executor.py

- **Role**: Enhanced command execution with AI optimization
- **Key Imports**:
  - `asyncio` for async execution
  - `subprocess` for process management
  - `server_core.*` modules for core services
- **Public Classes**:
  - `EnhancedCommandExecutor`: Main execution class
- **Public Functions**:
  - `execute_command()`: Executes security tools with optimization
  - `execute_commands_parallel()`: Executes multiple commands in parallel
- **Relationships**:
  - Used by MCP tools to execute security commands
  - Integrates with telemetry, caching, and monitoring
  - Handles command optimization and error recovery

### server_core/process_manager.py

- **Role**: Process management for security tool execution
- **Key Imports**:
  - `asyncio` for async operations
  - `subprocess` for process handling
- **Public Classes**:
  - `ProcessManager`: Manages tool execution processes
- **Public Functions**:
  - `run_command()`: Executes a command
  - `monitor_process()`: Monitors running processes
- **Relationships**:
  - Used by enhanced_command_executor.py
  - Manages process lifecycle and resource allocation

### server_core/resource_monitor.py

- **Role**: System resource monitoring
- **Key Imports**:
  - `psutil` for system monitoring
- **Public Classes**:
  - `ResourceMonitor`: Monitors CPU, memory, and disk usage
- **Public Functions**:
  - `get_current_usage()`: Gets current resource usage
  - `check_limits()`: Checks against resource limits
- **Relationships**:
  - Used by process manager and dashboard
  - Provides resource data for telemetry

### server_core/telemetry_collector.py

- **Role**: Telemetry data collection
- **Key Imports**:
  - `time` for timing operations
  - `collections` for data structures
- **Public Classes**:
  - `TelemetryCollector`: Collects and stores telemetry data
- **Public Functions**:
  - `record_execution()`: Records command execution metrics
  - `get_stats()`: Returns telemetry statistics
- **Relationships**:
  - Used by command executor and dashboard
  - Provides performance metrics for optimization

### server_core/cache.py

- **Role**: Caching system for command results
- **Key Imports**:
  - `time` for TTL management
  - `functools` for caching decorators
- **Public Classes**:
  - `HexStrikeCache`: Main caching system
- **Public Functions**:
  - `get()`: Retrieves cached data
  - `set()`: Stores data in cache
  - `get_stats()`: Returns cache statistics
- **Relationships**:
  - Used by command executor and other components
  - Reduces redundant command execution

### server_core/error_handling.py

- **Role**: Intelligent error handling and recovery
- **Key Imports**:
  - `logging` for error logging
  - `traceback` for error details
- **Public Classes**:
  - `IntelligentErrorHandler`: Handles errors with AI assistance
  - `GracefulDegradation`: Manages system degradation
- **Public Functions**:
  - `handle_error()`: Processes errors and suggests recovery
  - `degrade_services()`: Reduces functionality under load
- **Relationships**:
  - Used by command executor and other components
  - Provides error recovery capabilities

### server_core/intelligence/intelligent_decision_engine.py

- **Role**: AI-powered decision engine for security operations
- **Key Imports**:
  - `openai` for AI integration
  - `json` for data handling
- **Public Classes**:
  - `IntelligentDecisionEngine`: Main decision engine
- **Public Functions**:
  - `analyze_target()`: Analyzes target systems
  - `create_attack_chain()`: Creates attack strategies
  - `recommend_tools()`: Recommends security tools
- **Relationships**:
  - Used by MCP tools for intelligent operations
  - Integrates with vulnerability intelligence
  - Provides AI-powered recommendations

### server_core/vulnerability_correlator.py

- **Role**: Correlates vulnerability data from multiple sources
- **Key Imports**:
  - `sqlite3` for data storage
  - `json` for data handling
- **Public Classes**:
  - `VulnerabilityCorrelator`: Main correlation engine
- **Public Functions**:
  - `correlate_findings()`: Correlates vulnerability data
  - `get_related_vulnerabilities()`: Gets related vulnerabilities
- **Relationships**:
  - Used by decision engine and tools
  - Provides vulnerability context for operations

## Shared Modules

### shared/attack_chain.py

- **Role**: Attack chain modeling
- **Key Imports**:
  - `json` for data serialization
- **Public Classes**:
  - `AttackChain`: Represents attack sequences
- **Public Functions**:
  - `from_dict()`: Creates from dictionary
  - `to_dict()`: Converts to dictionary
- **Relationships**:
  - Used by decision engine and workflows
  - Provides attack structure modeling

### shared/target_profile.py

- **Role**: Target system profiling
- **Key Imports**:
  - `json` for data serialization
- **Public Classes**:
  - `TargetProfile`: Represents target system characteristics
- **Public Functions**:
  - `from_dict()`: Creates from dictionary
  - `to_dict()`: Converts to dictionary
- **Relationships**:
  - Used by decision engine and reconnaissance tools
  - Provides target context for operations

### shared/colored_formatter.py

- **Role**: Colored logging formatter
- **Key Imports**:
  - `logging` for logging framework
- **Public Classes**:
  - `ColoredFormatter`: Provides colored log output
- **Relationships**:
  - Used by main entry points for logging setup
  - Provides visual feedback for operations

## Skills Modules

Each skill directory contains specialized security workflows:

- **active-directory/**: Active Directory penetration testing skills
- **binary-analysis/**: Binary analysis and reverse engineering skills
- **cloud-audit/**: Cloud security auditing skills
- **exploitation/**: Exploitation framework skills
- **hexstrike-workflows/**: Custom HexStrike workflows
- **nmap-recon/**: Network reconnaissance skills
- **osint-recon/**: OSINT reconnaissance skills
- **password-cracking/**: Password cracking skills
- **smb-enum/**: SMB enumeration skills
- **subdomain-enum/**: Subdomain enumeration skills
- **web-recon/**: Web reconnaissance skills
- **web-vuln/**: Web vulnerability testing skills
- **wifi-pentest/**: WiFi penetration testing skills

Each skill directory contains:

- **REFERENCE.md**: Technical reference for the skill
- **SKILL.md**: Skill documentation and usage
- **Integration**: Skills integrate with the MCP framework and decision engine

## Test Modules

The tests directory contains comprehensive test coverage:

- **test_*.py**: Individual test files for each component
- **conftest.py**: Test configuration and fixtures
- **Integration tests**: End-to-end testing of workflows
- **Unit tests**: Component-level testing
- **Regression tests**: Preventing regressions in functionality

## UI Components

The ui directory contains a web-based dashboard:

- **src/**: TypeScript/React source code
- **public/**: Static assets
- **package.json**: Node.js dependencies
- **Dashboard**: Real-time monitoring and control interface

## MCP Tools Directory

The mcp_tools directory contains implementations for security tools organized by category:

- **active_directory/**: AD tools implementation
- **net_scan/**: Network scanning tools
- **web_scan/**: Web scanning tools
- **password_cracking/**: Password cracking tools
- **exploit_framework/**: Exploitation tools
- **osint/**: OSINT tools
- **wifi_pentest/**: WiFi testing tools
- **And many more categories**

Each tool implementation provides:

- Tool-specific execution logic
- Parameter validation and processing
- Output parsing and formatting
- Error handling and recovery

## Server API Modules

The server_api directory provides API endpoints for various functionalities:

- **Health monitoring**: System health checks
- **Tool execution**: Security tool execution endpoints
- **Data retrieval**: Configuration and telemetry data
- **Dashboard API**: Real-time data for the web interface

## Architecture Relationships

### Data Flow

1. **Entry Points**: hexstrike_mcp.py or hexstrike_server.py
2. **MCP Server**: mcp_core/mcp_entry.py initializes server
3. **Tool Registry**: tool_registry.py provides tool definitions
4. **Command Execution**: server_core/enhanced_command_executor.py
5. **Process Management**: server_core/process_manager.py
6. **Resource Monitoring**: server_core/resource_monitor.py
7. **Caching**: server_core/cache.py
8. **Telemetry**: server_core/telemetry_collector.py
9. **Error Handling**: server_core/error_handling.py

### Key Integration Points

- **MCP Tools**: Integrate with command executor through standardized interfaces
- **Decision Engine**: Provides AI-powered recommendations to tools
- **Singletons**: Central services accessed by all components
- **Skills**: Extend functionality through specialized workflows
- **UI**: Provides real-time monitoring and control

### External Dependencies

- **FastMCP**: MCP protocol implementation
- **Starlette**: Web framework for dashboard
- **OpenAI**: AI integration for decision engine
- **SQLite**: Local data storage
- **psutil**: System monitoring
- **Security Tools**: Integration with external security tools

## Development Patterns

### Module Organization

- **Core functionality** in server_core/
- **Tool integrations** in mcp_core/ and mcp_tools/
- **Shared utilities** in shared/
- **Specialized workflows** in skills/
- **Testing** in tests/
- **UI components** in ui/

### Code Structure

- **Entry points** with clear initialization
- **Singleton pattern** for shared services
- **Factory pattern** for tool creation
- **Strategy pattern** for different execution approaches
- **Observer pattern** for telemetry and monitoring

### Error Handling

- **Intelligent error recovery** with AI assistance
- **Graceful degradation** under load
- **Comprehensive logging** with colored output
- **Retry mechanisms** for transient failures

## Performance Considerations

- **Caching** reduces redundant command execution
- **Parallel execution** improves throughput
- **Resource monitoring** prevents system overload
- **Telemetry-driven optimization** improves performance over time
- **Lazy loading** of tools reduces startup time

## Security Considerations

- **Input validation** for all command parameters
- **Secure credential handling** for authentication
- **Sandboxed execution** for security tools
- **Access control** for sensitive operations
- **Audit logging** for compliance requirements

This breakdown provides a comprehensive view of the HexStrike AI codebase architecture, helping developers understand the relationships between modules and the overall system design.
