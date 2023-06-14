对于T个时序来的任务，对每个step的任务搜索最优参数，并且根据结果，决定下个step的任务运行，最终得到效果最好的一组。

接口需求：

1. 任务训练调度接口
   支持cmd传参（cuda、model input、model output、result file、r、epoch、lr、data dir、train bs、test bs、lamda_1、lamda_2）
2. 任务输出格式化接口
   a. model output path
   b. predict result (eval result, training params, model output)
3. 输出结果监控程序
   a. step起始标志
   b. 调参任务是否全部完成
4. GPU状态监控、任务提交程序
   a. 检测GPU显存占用，输出可用GPU
   b. 根据cmd pool，提交单卡任务


TODO

1. save average acc list	Done
2. check params   Done
3. modify lora_dim for script  Todo
