	?J?8??J@?J?8??J@!?J?8??J@	۝????۝????!۝????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?J?8??J@?k???JI@1?G??5\??A0?[w???I:d?w??Y??lY???rEagerKernelExecute 0*?ʡE???@)      @=2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?jdWZF??!????\?H@)??JU??1?:???E@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapB??	ܺ??!T??cY@C@)\??J?H??1?,?k??9@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatO?)??Y??!????f)@)???????1`x2m?'@:Preprocessing2U
Iterator::Model::ParallelMapV2??U????!Gȇ?9@)??U????1Gȇ?9@:Preprocessing2F
Iterator::Model<?$???!?Lҙ?a@)???@,???1????Q
@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?Rb????!?*]??@)?Rb????1?*]??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?)t^c???!?@??F	@)ԛQ?U??1s?|??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate? ?6qr??!	3?H) @)ge???\??1?????-??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????W??!??hjK@)&?fe???1_#;?mt??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor\Va3?y?!??????)\Va3?y?1??????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate[0]::TensorSliceI?"i7?x?!)f??;???)I?"i7?x?1)f??;???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?[???u?!?8?Q???)?[???u?1?8?Q???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate???G?Ȁ?!I{I(@??)?GnM?-a?1? ?C)???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensor??D-ͭP?!?|f?$??)??D-ͭP?1?|f?$??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 95.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ܝ????I???
??X@Qc?4?????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?k???JI@?k???JI@!?k???JI@      ??!       "	?G??5\???G??5\??!?G??5\??*      ??!       2	0?[w???0?[w???!0?[w???:	:d?w??:d?w??!:d?w??B      ??!       J	??lY?????lY???!??lY???R      ??!       Z	??lY?????lY???!??lY???b      ??!       JGPUYܝ????b q???
??X@yc?4?????