# Rule vs Qwen discovery comparison (LIBERO, local vLLM Qwen3-8B)

## episode 0
- instruction: 'turn on the stove and put the moka pot on it'
- rule: ['gripper', 'robot hand', 'stove and put moka pot on it']
- qwen: ['gripper', 'moka pot', 'robot hand', 'stove']

## episode 43
- instruction: 'put the black bowl in the bottom drawer of the cabinet and close it'
- rule: ['black bowl', 'black bowl in bottom drawer of cabinet and close it', 'bowl', 'drawer', 'gripper', 'robot hand']
- qwen: ['black bowl', 'bottom drawer', 'cabinet', 'gripper', 'robot hand']

## episode 78
- instruction: 'put the yellow and white mug in the microwave and close it'
- rule: ['gripper', 'robot hand', 'white mug', 'yellow and', 'yellow and white mug in microwave and close it']
- qwen: ['gripper', 'microwave', 'robot hand', 'yellow and white mug']

## episode 110
- instruction: 'put both moka pots on the stove'
- rule: ['gripper', 'robot hand', 'stove']
- qwen: ['gripper', 'moka pot', 'robot hand', 'stove']

## episode 143
- instruction: 'put both the alphabet soup and the cream cheese box in the basket'
- rule: ['alphabet soup and cream cheese box in basket', 'box', 'gripper', 'robot hand']
- qwen: ['alphabet soup', 'basket', 'cream cheese box', 'gripper', 'robot hand']
