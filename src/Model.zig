const c = @import("c.zig");

state: ?*anyopaque,
prompt_fn: *const fn (token_id: i32) callconv(.C) bool,
response_fn: *const fn (
    token_id: i32,
    response: [*c]const u8,
) callconv(.C) bool,
recalculate_fn: *const fn (is_recalculating: bool) callconv(.C) bool,

const Model = @This();

const PromptOption = struct {
    prompt: []const u8,
    m: ?*anyopaque,
    result: []u8,
    ctx: c.llmodel_prompt_context = .{
        .logits = null,
        .logits_size = 0,
        .tokens = null,
        .tokens_size = 0,
        .n_past = 0,
        .n_ctx = 1024,
        .n_predict = 50,
        .top_k = 10,
        .top_p = 0.9,
        .temp = 1.0,
        .n_batch = 1,
        .repeat_penalty = 1.2,
        .repeat_last_n = 10,
        .context_erase = 0.5,
    },
};

const InitOptions = struct {
    path: [*c]const u8,
    threads: i32,
    mtype: enum {
        llama,
        gptj,
        mpt,
    },
    prompt_fn: *const fn (token_id: i32) callconv(.C) bool,
    response_fn: *const fn (
        token_id: i32,
        response: [*c]const u8,
    ) callconv(.C) bool,
    recalculate_fn: *const fn (is_recalculating: bool) callconv(.C) bool,
};

pub fn init(options: InitOptions) !Model {
    var gptj = blk: {
        switch (options.mtype) {
            .mpt => break :blk c.llmodel_mpt_create(),
            .llama => break :blk c.llmodel_llama_create(),
            .gptj => break :blk c.llmodel_gptj_create(),
        }
    };

    c.llmodel_setThreadCount(gptj, options.threads);
    if (!c.llmodel_loadModel(gptj, options.path)) {
        return error.LoadModelFailed;
    }

    return .{
        .state = gptj,
        .prompt_fn = options.prompt_fn,
        .response_fn = options.response_fn,
        .recalculate_fn = options.recalculate_fn,
    };
}

pub fn deinit(model: *Model) void {
    var ctx = @ptrCast(c.llmodel_model, model.state);
    c.llmodel_llama_destroy(ctx);
}
