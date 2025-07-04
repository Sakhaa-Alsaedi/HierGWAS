#!/bin/bash

# HierGWAS Automated Setup and Execution Script
# This script automatically sets up and runs HierGWAS

set -e  # Exit on any error

echo "🚀 HierGWAS Automated Setup and Execution"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
        
        # Check if version is >= 3.8
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python version is compatible (>= 3.8)"
        else
            print_error "Python version must be >= 3.8. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check if we're in the HierGWAS directory
check_directory() {
    print_status "Checking directory structure..."
    if [[ ! -f "setup.py" ]] || [[ ! -d "hiergwas" ]]; then
        print_error "Please run this script from the HierGWAS root directory"
        print_error "Make sure you have setup.py and hiergwas/ directory"
        exit 1
    fi
    print_success "Directory structure looks good"
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [[ -d "hiergwas_env" ]]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf hiergwas_env
    fi
    
    python3 -m venv hiergwas_env
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source hiergwas_env/bin/activate
    print_success "Virtual environment activated"
}

# Install PyTorch
install_pytorch() {
    print_status "Installing PyTorch..."
    
    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_success "PyTorch installed"
}

# Install PyTorch Geometric
install_pyg() {
    print_status "Installing PyTorch Geometric..."
    
    pip install torch-geometric
    pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
    
    print_success "PyTorch Geometric installed"
}

# Install other dependencies
install_dependencies() {
    print_status "Installing other dependencies..."
    
    pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm
    
    print_success "Dependencies installed"
}

# Install HierGWAS
install_hiergwas() {
    print_status "Installing HierGWAS..."
    
    pip install -e .
    
    print_success "HierGWAS installed"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    python3 -c "
import hiergwas
from hiergwas import HierGWASModel, HierGWASConfig
from hiergwas.data import GWASData
from hiergwas.utils import create_synthetic_gwas_data
print('✅ All imports successful')
"
    
    print_success "Installation verified"
}

# Run basic example
run_basic_example() {
    print_status "Running basic usage example..."
    
    if [[ -f "examples/basic_usage.py" ]]; then
        python3 examples/basic_usage.py
        print_success "Basic example completed"
    else
        print_error "examples/basic_usage.py not found"
        exit 1
    fi
}

# Run advanced example (optional)
run_advanced_example() {
    echo ""
    read -p "Do you want to run the advanced example? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Running advanced usage example..."
        
        if [[ -f "examples/advanced_usage.py" ]]; then
            python3 examples/advanced_usage.py --use-synthetic --skip-hpo --skip-cv
            print_success "Advanced example completed"
        else
            print_error "examples/advanced_usage.py not found"
        fi
    else
        print_status "Skipping advanced example"
    fi
}

# Show results
show_results() {
    print_status "Checking generated files..."
    
    echo ""
    echo "📁 Generated Files:"
    
    if [[ -f "best_model.pth" ]]; then
        echo "  ✅ best_model.pth - Trained model weights"
    fi
    
    if [[ -d "hiergwas_model" ]]; then
        echo "  ✅ hiergwas_model/ - Complete model package"
    fi
    
    if [[ -f "attention_analysis.png" ]]; then
        echo "  ✅ attention_analysis.png - Attention visualizations"
    fi
    
    if [[ -f "training_curves.png" ]]; then
        echo "  ✅ training_curves.png - Training progress plots"
    fi
    
    if [[ -f "biological_analysis.pkl" ]]; then
        echo "  ✅ biological_analysis.pkl - Biological relevance analysis"
    fi
    
    if [[ -d "data" ]]; then
        echo "  ✅ data/ - Synthetic GWAS data"
    fi
    
    echo ""
}

# Cleanup function
cleanup() {
    print_status "Cleaning up temporary files..."
    # Add any cleanup commands here if needed
}

# Main execution
main() {
    echo "Starting HierGWAS setup and execution..."
    echo ""
    
    # Setup steps
    check_python
    check_directory
    create_venv
    activate_venv
    install_pytorch
    install_pyg
    install_dependencies
    install_hiergwas
    verify_installation
    
    echo ""
    print_success "🎉 Setup completed successfully!"
    echo ""
    
    # Execution steps
    run_basic_example
    run_advanced_example
    
    echo ""
    show_results
    
    echo ""
    print_success "🎉 HierGWAS execution completed!"
    echo ""
    echo "📖 Next Steps:"
    echo "  1. Explore the generated files and visualizations"
    echo "  2. Check examples/basic_usage.py to understand the code"
    echo "  3. Try modifying the configuration parameters"
    echo "  4. Use your own GWAS data instead of synthetic data"
    echo ""
    echo "📚 Documentation:"
    echo "  - HierGWAS_README.md - Complete usage guide"
    echo "  - QUICK_START_GUIDE.md - Quick start instructions"
    echo "  - INSTALLATION.md - Detailed installation guide"
    echo ""
    echo "🔧 To reactivate the environment later:"
    echo "  source hiergwas_env/bin/activate"
    echo ""
}

# Handle interruption
trap cleanup EXIT

# Parse command line arguments
SKIP_SETUP=false
SKIP_EXAMPLES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-examples)
            SKIP_EXAMPLES=true
            shift
            ;;
        --help|-h)
            echo "HierGWAS Automated Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-setup     Skip environment setup (use existing environment)"
            echo "  --skip-examples  Skip running examples"
            echo "  --help, -h       Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
if [[ "$SKIP_SETUP" == true ]]; then
    print_status "Skipping setup, using existing environment..."
    activate_venv
    verify_installation
    
    if [[ "$SKIP_EXAMPLES" == false ]]; then
        run_basic_example
        run_advanced_example
    fi
    
    show_results
else
    main
fi

