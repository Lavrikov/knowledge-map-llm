# Here we create example database with topic related questions and logs
# don't worry to execute it multiple times, this UNIQUE(..) flag will filter duplicates
# to change table structure better to remove database file and recreate it
import sqlite3

# 1. Connect or create a new database
connection = sqlite3.connect("./data/log_database.db")

# 2. Cursor to run SQL-requests
cursor = connection.cursor()

# 3. Create database tables if not exist

# Topics with descriptions
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Topic_tags (
        topic_tag TEXT NOT NULL,
        description TEXT NOT NULL,
        UNIQUE(topic_tag)
    )
""")

# Hight level questions attributed to a topic tag
cursor.execute("""
    CREATE TABLE IF NOT EXISTS GeneralQuestions (
        question_id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic_tag TEXT NOT NULL,
        question_text TEXT NOT NULL,
        UNIQUE(question_text)
    )
""")


# Sub level questions for every question_id
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Subquestions (
        subquestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_id INTEGER,
        topic_tag TEXT NOT NULL,
        question_text TEXT NOT NULL,
        UNIQUE(question_text) 
)
""")


# Raw logs texts
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Logs_texts (
        Logs_id INTEGER PRIMARY KEY AUTOINCREMENT,
        Logs_text TEXT NOT NULL,
        UNIQUE(Logs_text) 
    )
""")


print("Dataset created!")

# 4. Insert data, don't worry to execute it more than once UNIQUE(..) flag will filter duplicates

cursor.execute("INSERT OR IGNORE INTO Topic_tags (topic_tag, description) VALUES (?, ?)", ("NOFILE", "No file errors type"))

cursor.execute("INSERT OR IGNORE INTO GeneralQuestions (topic_tag, question_id, question_text) VALUES (?, ?, ?)", ("NOFILE", 1, "binary library file missing like .lib .dll ."))
cursor.execute("INSERT OR IGNORE INTO GeneralQuestions (topic_tag, question_id, question_text) VALUES (?, ?, ?)", ("NOFILE", 2, "project file missing like .py .js ."))
cursor.execute("INSERT OR IGNORE INTO GeneralQuestions (topic_tag, question_id, question_text) VALUES (?, ?, ?)", ("NOFILE", 3, "path to missing file looks like system path"))
cursor.execute("INSERT OR IGNORE INTO GeneralQuestions (topic_tag, question_id, question_text) VALUES (?, ?, ?)", ("NOFILE", 4, "path to missing file looks like relational project path"))
cursor.execute("INSERT OR IGNORE INTO GeneralQuestions (topic_tag, question_id, question_text) VALUES (?, ?, ?)", ("NOFILE", 5, "path to missing file looks like path to virtual env or conda"))

cursor.execute("INSERT OR IGNORE INTO Logs_texts (Logs_id, Logs_text) VALUES (?, ?)", (1, """
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
[<ipython-input-8-3e6b92348d53>](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in <cell line: 1>()
----> 1 audio = pipe("brazilian samba drums").audios[0]

52 frames
[/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in decorate_context(*args, **kwargs)
    113     def decorate_context(*args, **kwargs):
    114         with ctx_factory():
--> 115             return func(*args, **kwargs)
    116 
    117     return decorate_context

[/usr/local/lib/python3.10/dist-packages/diffusers/pipelines/audioldm2/pipeline_audioldm2.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in __call__(self, prompt, audio_length_in_s, num_inference_steps, guidance_scale, negative_prompt, num_waveforms_per_prompt, eta, generator, latents, prompt_embeds, negative_prompt_embeds, generated_prompt_embeds, negative_generated_prompt_embeds, attention_mask, negative_attention_mask, max_new_tokens, return_dict, callback, callback_steps, cross_attention_kwargs, output_type)
    925 
    926                 # predict the noise residual
--> 927                 noise_pred = self.unet(
    928                     latent_model_input,
    929                     t,

[/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in _wrapped_call_impl(self, *args, **kwargs)
   1516             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1517         else:
-> 1518             return self._call_impl(*args, **kwargs)
   1519 
   1520     def _call_impl(self, *args, **kwargs):

[/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in _call_impl(self, *args, **kwargs)
   1525                 or _global_backward_pre_hooks or _global_backward_hooks
   1526                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1527             return forward_call(*args, **kwargs)
   1528 
   1529         try:

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/eval_frame.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in _fn(*args, **kwargs)
    326             dynamic_ctx.__enter__()
    327             try:
--> 328                 return fn(*args, **kwargs)
    329             finally:
    330                 set_eval_frame(prior)

[/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in _wrapped_call_impl(self, *args, **kwargs)
   1516             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1517         else:
-> 1518             return self._call_impl(*args, **kwargs)
   1519 
   1520     def _call_impl(self, *args, **kwargs):

[/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in _call_impl(self, *args, **kwargs)
   1525                 or _global_backward_pre_hooks or _global_backward_hooks
   1526                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1527             return forward_call(*args, **kwargs)
   1528 
   1529         try:

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/eval_frame.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in catch_errors(frame, cache_entry, frame_state)
    486 
    487         with compile_lock, _disable_current_modes():
--> 488             return callback(frame, cache_entry, hooks, frame_state)
    489 
    490     catch_errors._torchdynamo_orig_callable = callback  # type: ignore[attr-defined]

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in _convert_frame(frame, cache_entry, hooks, frame_state)
    623         counters["frames"]["total"] += 1
    624         try:
--> 625             result = inner_convert(frame, cache_entry, hooks, frame_state)
    626             counters["frames"]["ok"] += 1
    627             return result

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in _fn(*args, **kwargs)
    137         cleanup = setup_compile_debug()
    138         try:
--> 139             return fn(*args, **kwargs)
    140         finally:
    141             cleanup.close()

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in _convert_frame_assert(frame, cache_entry, hooks, frame_state)
    378         )
    379 
--> 380         return _compile(
    381             frame.f_code,
    382             frame.f_globals,

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in _compile(code, globals, locals, builtins, compiler_fn, one_graph, export, export_constraints, hooks, cache_size, frame, frame_state, compile_id)
    553     with compile_context(CompileContext(compile_id)):
    554         try:
--> 555             guarded_code = compile_inner(code, one_graph, hooks, transform)
    556             return guarded_code
    557         except (

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/utils.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in time_wrapper(*args, **kwargs)
    187             with torch.profiler.record_function(f"{key} (dynamo_timed)"):
    188                 t0 = time.time()
--> 189                 r = func(*args, **kwargs)
    190                 time_spent = time.time() - t0
    191             compilation_time_metrics[key].append(time_spent)

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in compile_inner(code, one_graph, hooks, transform)
    475         for attempt in itertools.count():
    476             try:
--> 477                 out_code = transform_code_object(code, transform)
    478                 orig_code_map[out_code] = code
    479                 break

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/bytecode_transformation.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in transform_code_object(code, transformations, safe)
   1026     propagate_line_nums(instructions)
   1027 
-> 1028     transformations(instructions, code_options)
   1029     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1030 

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in transform(instructions, code_options)
    442         try:
    443             with tracing(tracer.output.tracing_context):
--> 444                 tracer.run()
    445         except (exc.RestartAnalysis, exc.SkipFrame):
    446             raise

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in run(self)
   2072 
   2073     def run(self):
-> 2074         super().run()
   2075 
   2076     def match_nested_cell(self, name, cell):

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in run(self)
    722                     self.instruction_pointer is not None
    723                     and not self.output.should_exit
--> 724                     and self.step()
    725                 ):
    726                     pass

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in step(self)
    686                 self.f_code.co_filename, self.lineno, self.f_code.co_name
    687             )
--> 688             getattr(self, inst.opname)(inst)
    689 
    690             return inst.opname != "RETURN_VALUE"

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/symbolic_convert.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in RETURN_VALUE(self, inst)
   2160         )
   2161         log.debug("RETURN_VALUE triggered compile")
-> 2162         self.output.compile_subgraph(
   2163             self,
   2164             reason=GraphCompileReason(

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in compile_subgraph(self, tx, partial_convert, reason)
    855             if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
    856                 output.extend(
--> 857                     self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
    858                 )
    859 

[/usr/lib/python3.10/contextlib.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in inner(*args, **kwds)
     77         def inner(*args, **kwds):
     78             with self._recreate_cm():
---> 79                 return func(*args, **kwds)
     80         return inner
     81 

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in compile_and_call_fx_graph(self, tx, rv, root)
    955         )
    956 
--> 957         compiled_fn = self.call_user_compiler(gm)
    958         compiled_fn = disable(compiled_fn)
    959 

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/utils.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in time_wrapper(*args, **kwargs)
    187             with torch.profiler.record_function(f"{key} (dynamo_timed)"):
    188                 t0 = time.time()
--> 189                 r = func(*args, **kwargs)
    190                 time_spent = time.time() - t0
    191             compilation_time_metrics[key].append(time_spent)

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in call_user_compiler(self, gm)
   1022             unimplemented_with_warning(e, self.root_tx.f_code, msg)
   1023         except Exception as e:
-> 1024             raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
   1025                 e.__traceback__
   1026             ) from None

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/output_graph.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in call_user_compiler(self, gm)
   1007             if config.verify_correctness:
   1008                 compiler_fn = WrapperBackend(compiler_fn)
-> 1009             compiled_fn = compiler_fn(gm, self.example_inputs())
   1010             _step_logger()(logging.INFO, f"done compiler function {name}")
   1011             assert callable(compiled_fn), "compiler_fn did not return callable"

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/repro/after_dynamo.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in debug_wrapper(gm, example_inputs, **kwargs)
    115                     raise
    116         else:
--> 117             compiled_gm = compiler_fn(gm, example_inputs)
    118 
    119         return compiled_gm

[/usr/local/lib/python3.10/dist-packages/torch/__init__.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in __call__(self, model_, inputs_)
   1566         from torch._inductor.compile_fx import compile_fx
   1567 
-> 1568         return compile_fx(model_, inputs_, config_patches=self.config)
   1569 
   1570     def get_compiler_config(self):

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1148         tracing_context
   1149     ), compiled_autograd.disable():
-> 1150         return aot_autograd(
   1151             fw_compiler=fw_compiler,
   1152             bw_compiler=bw_compiler,

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/backends/common.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in compiler_fn(gm, example_inputs)
     53             # NB: NOT cloned!
     54             with enable_aot_logging(), patch_config:
---> 55                 cg = aot_module_simplified(gm, example_inputs, **kwargs)
     56                 counters["aot_autograd"]["ok"] += 1
     57                 return disable(cg)

[/usr/local/lib/python3.10/dist-packages/torch/_functorch/aot_autograd.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations, inference_compiler)
   3889 
   3890     with compiled_autograd.disable():
-> 3891         compiled_fn = create_aot_dispatcher_function(
   3892             functional_call,
   3893             full_args,

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/utils.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in time_wrapper(*args, **kwargs)
    187             with torch.profiler.record_function(f"{key} (dynamo_timed)"):
    188                 t0 = time.time()
--> 189                 r = func(*args, **kwargs)
    190                 time_spent = time.time() - t0
    191             compilation_time_metrics[key].append(time_spent)

[/usr/local/lib/python3.10/dist-packages/torch/_functorch/aot_autograd.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in create_aot_dispatcher_function(flat_fn, flat_args, aot_config)
   3427         # You can put more passes here
   3428 
-> 3429         compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config, fw_metadata=fw_metadata)
   3430         if aot_config.is_export:
   3431 

[/usr/local/lib/python3.10/dist-packages/torch/_functorch/aot_autograd.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in aot_wrapper_dedupe(flat_fn, flat_args, aot_config, compiler_fn, fw_metadata)
   2210 
   2211     if ok:
-> 2212         return compiler_fn(flat_fn, leaf_flat_args, aot_config, fw_metadata=fw_metadata)
   2213 
   2214     # export path: ban duplicate inputs for now, add later if requested.

[/usr/local/lib/python3.10/dist-packages/torch/_functorch/aot_autograd.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in aot_wrapper_synthetic_base(flat_fn, flat_args, aot_config, fw_metadata, needs_autograd, compiler_fn)
   2390     # Happy path: we don't need synthetic bases
   2391     if synthetic_base_info is None:
-> 2392         return compiler_fn(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)
   2393 
   2394     # export path: ban synthetic bases for now, add later if requested.

[/usr/local/lib/python3.10/dist-packages/torch/_functorch/aot_autograd.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in aot_dispatch_base(flat_fn, flat_args, aot_config, fw_metadata)
   1571         if torch._guards.TracingContext.get():
   1572             torch._guards.TracingContext.get().fw_metadata = fw_metadata
-> 1573         compiled_fw = compiler(fw_module, flat_args)
   1574 
   1575     # This boxed_call handling happens inside create_runtime_wrapper as well.

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/utils.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in time_wrapper(*args, **kwargs)
    187             with torch.profiler.record_function(f"{key} (dynamo_timed)"):
    188                 t0 = time.time()
--> 189                 r = func(*args, **kwargs)
    190                 time_spent = time.time() - t0
    191             compilation_time_metrics[key].append(time_spent)

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in fw_compiler_base(model, example_inputs, is_inference)
   1090             }
   1091 
-> 1092         return inner_compile(
   1093             model,
   1094             example_inputs,

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/repro/after_aot.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in debug_wrapper(gm, example_inputs, **kwargs)
     78             # Call the compiler_fn - which is either aot_autograd or inductor
     79             # with fake inputs
---> 80             inner_compiled_fn = compiler_fn(gm, example_inputs)
     81         except Exception as e:
     82             # TODO: Failures here are troublesome because no real inputs,

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/debug.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in inner(*args, **kwargs)
    226         def inner(*args, **kwargs):
    227             with DebugContext():
--> 228                 return fn(*args, **kwargs)
    229 
    230         return wrap_compiler_debug(inner, compiler_name="inductor")

[/usr/lib/python3.10/contextlib.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in inner(*args, **kwds)
     77         def inner(*args, **kwds):
     78             with self._recreate_cm():
---> 79                 return func(*args, **kwds)
     80         return inner
     81 

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in newFunction(*args, **kwargs)
     52             @wraps(old_func)
     53             def newFunction(*args, **kwargs):
---> 54                 return old_func(*args, **kwargs)
     55 
     56             return newFunction

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in compile_fx_inner(gm, example_inputs, cudagraphs, num_fixed, is_backward, graph_id, cpp_wrapper, aot_mode, is_inference, boxed_forward_device_index, user_visible_outputs, layout_opt)
    339     }
    340 
--> 341     compiled_graph: CompiledFxGraph = fx_codegen_and_compile(
    342         *graph_args, **graph_kwargs  # type: ignore[arg-type]
    343     )

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_fx.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in fx_codegen_and_compile(gm, example_inputs, cudagraphs, num_fixed, is_backward, graph_id, cpp_wrapper, aot_mode, is_inference, user_visible_outputs, layout_opt)
    563                     else:
    564                         context.output_strides.append(None)
--> 565             compiled_fn = graph.compile_to_fn()
    566 
    567             if graph.disable_cudagraphs:

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/graph.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in compile_to_fn(self)
    965             return AotCodeCache.compile(self, code, cuda=self.cuda)
    966         else:
--> 967             return self.compile_to_module().call
    968 
    969     def get_output_names(self):

[/usr/local/lib/python3.10/dist-packages/torch/_dynamo/utils.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in time_wrapper(*args, **kwargs)
    187             with torch.profiler.record_function(f"{key} (dynamo_timed)"):
    188                 t0 = time.time()
--> 189                 r = func(*args, **kwargs)
    190                 time_spent = time.time() - t0
    191             compilation_time_metrics[key].append(time_spent)

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/graph.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in compile_to_module(self)
    936         linemap = [(line_no, node.stack_trace) for line_no, node in linemap]
    937         key, path = PyCodeCache.write(code)
--> 938         mod = PyCodeCache.load_by_key_path(key, path, linemap=linemap)
    939         self.cache_key = key
    940         self.cache_path = path

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/codecache.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in load_by_key_path(cls, key, path, linemap)
   1137                 mod.__file__ = path
   1138                 mod.key = key
-> 1139                 exec(code, mod.__dict__, mod.__dict__)
   1140                 sys.modules[mod.__name__] = mod
   1141                 # another thread might set this first

[/tmp/torchinductor_root/7n/c7nibse6wgoaypjbdhgrkgsgbfcxqvdo7jque3pr52vtf6vor5yi.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in <module>
   7289 
   7290 
-> 7291 async_compile.wait(globals())
   7292 del async_compile
   7293 

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/codecache.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in wait(self, scope)
   1416                     pbar.set_postfix_str(key)
   1417                 if isinstance(result, (Future, TritonFuture)):
-> 1418                     scope[key] = result.result()
   1419                     pbar.update(1)
   1420 

[/usr/local/lib/python3.10/dist-packages/torch/_inductor/codecache.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in result(self)
   1275             return self.kernel
   1276         # If the worker failed this will throw an exception.
-> 1277         self.future.result()
   1278         kernel = self.kernel = _load_kernel(self.kernel_name, self.source_code)
   1279         latency = time() - t0

[/usr/lib/python3.10/concurrent/futures/_base.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in result(self, timeout)
    456                     raise CancelledError()
    457                 elif self._state == FINISHED:
--> 458                     return self.__get_result()
    459                 else:
    460                     raise TimeoutError()

[/usr/lib/python3.10/concurrent/futures/_base.py](https://f1ggesi86gr-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab-20230823-060135-RC00_559378898#) in __get_result(self)
    401         if self._exception:
    402             try:
--> 403                 raise self._exception
    404             finally:
    405                 # Break a reference cycle with the exception in self._exception

BackendCompilerFailed: backend='inductor' raised:
AssertionError: libcuda.so cannot found!


Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
"""
))

print("Data added!")

# 5. Save new insertion to database
connection.commit()

# 6. Check with request SELECT
cursor.execute("SELECT * FROM GeneralQuestions")
rows = cursor.fetchall()

print("Test GeneralQuestions data:")
for row in rows:
    print(row)

cursor.execute("SELECT * FROM Topic_tags")
rows = cursor.fetchall()

print("Test Topic_tags data:")
for row in rows:
    print(row)

cursor.execute("SELECT * FROM Logs_texts")
rows = cursor.fetchall()

print("Test Logs_texts data:")
for row in rows:
    print(row)

# 7. Deattach from database
connection.close()
