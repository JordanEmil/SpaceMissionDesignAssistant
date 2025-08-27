#!/bin/bash

# RAG Evaluation Pipeline Runner Script
# Makes it easy to run different evaluation configurations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY environment variable not set${NC}"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 [OPTION]"
    echo "Options:"
    echo "  quick       Run quick test evaluation (minimal parameters)"
    echo "  full        Run full evaluation with all parameters"
    echo "  generate    Only generate synthetic data"
    echo "  evaluate    Only run evaluation (skip data generation)"
    echo "  visualize   Only create visualizations (requires timestamp)"
    echo "  help        Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick                    # Run quick test"
    echo "  $0 full                     # Run full evaluation"
    echo "  $0 visualize 20240101_120000  # Create visualizations for specific results"
}

# Parse command line arguments
case "$1" in
    quick)
        echo -e "${GREEN}Running Quick Test Evaluation...${NC}"
        python run_evaluation_pipeline.py --quick-test
        ;;
    
    full)
        echo -e "${GREEN}Running Full Evaluation Pipeline...${NC}"
        echo -e "${YELLOW}Warning: This will take several hours and use significant API calls${NC}"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python run_evaluation_pipeline.py
        else
            echo "Aborted."
        fi
        ;;
    
    generate)
        echo -e "${GREEN}Generating Synthetic Data Only...${NC}"
        cd scripts
        python generate_synthetic_data.py
        cd ..
        ;;
    
    evaluate)
        echo -e "${GREEN}Running Evaluation Only...${NC}"
        python run_evaluation_pipeline.py --skip-data-generation
        ;;
    
    visualize)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Timestamp required for visualization${NC}"
            echo "Usage: $0 visualize <timestamp>"
            echo "Example: $0 visualize 20240101_120000"
            exit 1
        fi
        echo -e "${GREEN}Creating Visualizations for timestamp: $2${NC}"
        cd scripts
        python visualization_and_reporting.py "$2"
        cd ..
        ;;
    
    help|--help|-h)
        usage
        ;;
    
    *)
        echo -e "${RED}Invalid option: $1${NC}"
        usage
        exit 1
        ;;
esac

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Operation completed successfully!${NC}"
    
    # Show where to find results
    if [ "$1" != "help" ]; then
        echo ""
        echo "Results locations:"
        echo "  - Evaluation data: evaluation/data/"
        echo "  - Results: evaluation/results/"
        echo "  - Visualizations: evaluation/plots/"
        echo "  - Report: evaluation/plots/evaluation_report.html"
    fi
else
    echo -e "${RED}✗ Operation failed${NC}"
    exit 1
fi