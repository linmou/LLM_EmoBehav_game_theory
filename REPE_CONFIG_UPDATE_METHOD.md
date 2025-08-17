# repe_config Update Method

## 📝 **Overview**

Added a new method `update_repe_config_from_yaml()` that allows both `experiment_series_runner.py` and `game_theory_exp_0205.py` to update `repe_config` directly from YAML configuration files.

## 🔧 **Changes Made**

### **1. New Functions in `experiment_config.py`:**

#### **`update_repe_config_from_yaml(base_repe_config, yaml_config)`**
- Updates repe_config with values from YAML config
- Supports multiple configuration sources with priority ordering
- Provides clear logging of applied updates

#### **Enhanced `get_repe_eng_config(model_name, yaml_config_path=None, yaml_config=None)`**
- Backward compatible: works exactly as before when called with just `model_name`
- New optional parameters for YAML config integration
- Maintains all existing functionality

### **2. Updated Integration Points:**

#### **`experiment_series_runner.py`**
```python
# Before
repe_eng_config = get_repe_eng_config(model_name)

# After  
repe_eng_config = get_repe_eng_config(model_name, yaml_config=exp_config)
```

#### **`game_theory_exp_0205.py`**
```python
# Before
repe_eng_config = get_repe_eng_config(model_name)

# After
repe_eng_config = get_repe_eng_config(model_name, yaml_config=exp_config)
```

## 📊 **Configuration Priority (Highest to Lowest)**

### **1. Direct `repe_config` Section (Highest Priority)**
```yaml
repe_config:
  multimodal_intent: true      # Direct override
  data_dir: "data/image"       # Direct override
  emotions: ["anger", "happy"] # Direct override
  rep_token: -1               # Direct override
```

### **2. Experiment-Level Mappings (Fallback)**
```yaml
experiment:
  emotions: ["anger", "happiness", "sadness"]  # → repe_config.emotions
  data:
    data_dir: "data/text"                      # → repe_config.data_dir  
  llm:
    model_name: "Qwen/Qwen2.5-VL-3B-Instruct" # → repe_config.model_name_or_path
```

### **3. Base Configuration (Default)**
- Original hardcoded values in `get_repe_eng_config()`

## 🎯 **Usage Examples**

### **For Multimodal Experiments:**
```yaml
# config/qwen2.5_MM_Series_Prisoners_Dilemm.yaml
experiment:
  name: "Multimodal_Experiment"
  # ... other experiment config ...

# Add this section for multimodal processing
repe_config:
  multimodal_intent: true     # 🔑 Key field for multimodal
  data_dir: "data/image"      # Points directly to image JSONs
  
  # Optional overrides
  emotions: ["anger", "happiness", "sadness", "fear"]
  rep_token: -1
  direction_method: "pca"
  rebuild: false
```

### **Running the Experiment:**
```bash
# experiment_series_runner.py
python neuro_manipulation/experiment_series_runner.py --config config/qwen2.5_MM_Series_Prisoners_Dilemm.yaml

# game_theory_exp_0205.py  
python neuro_manipulation/game_theory_exp_0205.py --config config/qwen2.5_MM_Series_Prisoners_Dilemm.yaml
```

## 📈 **Benefits**

### **1. Unified Configuration**
- Single YAML file controls both experiment and repe settings
- No need to modify Python code for configuration changes
- Clear separation of concerns with priority system

### **2. Multimodal Support**
- Easy enablement of multimodal processing via `multimodal_intent: true`
- Direct control over data directory and processing parameters
- Seamless integration with existing auto-detection system

### **3. Backward Compatibility**
- All existing code continues to work unchanged
- No breaking changes to function signatures
- Optional parameters maintain default behavior

### **4. Developer Experience**
- Clear logging shows which values are being updated
- Flexible configuration sources (file path vs. pre-loaded config)
- Comprehensive error handling and validation

## 🔍 **Logging Output**

When config updates are applied, you'll see:
```
✓ Updated repe_config.multimodal_intent = True
✓ Updated repe_config.data_dir = data/image
✓ Updated repe_config.emotions = ['anger', 'happiness', 'sadness', 'fear']
✓ Updated model_name_or_path from experiment.llm.model_name = Qwen/Qwen2.5-VL-3B-Instruct
```

## 🚀 **Complete Example Config**

```yaml
experiment:
  name: "My_Multimodal_Experiment"
  games: ["Prisoners_Dilemma"]
  models: ["Qwen/Qwen2.5-VL-3B-Instruct"]
  
  game:
    name: "Prisoners_Dilemma"
  
  llm:
    model_name: 'Qwen/Qwen2.5-VL-3B-Instruct'
    
  emotions:
    - "anger"
    - "happiness"
    - "sadness"
    
  output:
    base_dir: "results/my_experiment"

# Essential for multimodal processing
repe_config:
  multimodal_intent: true  # Enables multimodal processing
  data_dir: "data/image"   # Points to image emotion JSONs
  rebuild: false           # Use cached emotion readers
```

## ✅ **Testing**

The functionality has been thoroughly tested with:
- ✅ YAML config updates working correctly
- ✅ Backward compatibility maintained  
- ✅ File path and pre-loaded config support
- ✅ Priority system functioning as designed
- ✅ Clear logging and error handling

**Ready for production use!** 🎉