	?	??$?A@?	??$?A@!?	??$?A@	??? |????? |??!??? |??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?	??$?A@?>tA}??@1?`S?Q???AM֨???Ih!???@Ya?
?+???rEagerKernelExecute 0*	?ʡE?_}@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapx?=\r??!?(?N?VK@)??kЗ???1ot?{"UF@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?[?J???! RyNQ:@)???=zñ?1TS?B?-@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?7iͫ?!???Z'@)?% ??*??1?t^???$@:Preprocessing2U
Iterator::Model::ParallelMapV2_@/ܹ0??![??n?<@)_@/ܹ0??1[??n?<@:Preprocessing2F
Iterator::Model?w?-;į?!_k?7g*@)oH?'??1dBS?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateO@a?ӛ?!	Y? @)Oʤ?6 ??1??q@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat.?熦???!???,d@)ĖM?d??1???m"C	@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch[%XΌ?!?Z???@)[%XΌ?1?Z???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??x?&1??!|x??=N@)???? |?1??缺E??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/kb???v?!#c?:m
??)/kb???v?1#c?:m
??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??H??u?!?W?#ۄ??)??H??u?1?W?#ۄ??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate[0]::TensorSliceSv?A]?p?!MDh????)Sv?A]?p?1MDh????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate??9??w?!3@a???)ςP???\?1?f?????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensor???M?qJ?!??~????)???M?qJ?1??~????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 90.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??? |??I@D?l?X@Q^f?M??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?>tA}??@?>tA}??@!?>tA}??@      ??!       "	?`S?Q????`S?Q???!?`S?Q???*      ??!       2	M֨???M֨???!M֨???:	h!???@h!???@!h!???@B      ??!       J	a?
?+???a?
?+???!a?
?+???R      ??!       Z	a?
?+???a?
?+???!a?
?+???b      ??!       JGPUY??? |??b q@D?l?X@y^f?M??