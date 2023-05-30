const c = @import("c.zig");

state: ?*anyopaque,

const Model = @This();

const PromptOption = struct {
    prompt: []const u8,
    m: ?*anyopaque,
    result: []u8,
    prompt: c.llmodel_prompt_context = .{
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
};

pub fn init(options: InitOptions) Model {
    var gptj = blk: {
        switch (options.mtype) {
            .mpt => break :blk c.llmodel_mpt_create(),
            .llama => break :blk c.llmodel_llama_create(),
            .gptj => break :blk c.llmodel_gptj_create(),
        }
    };

    c.llmodel_setThreadCount(gptj, options.threads);
    if (!c.llmodel_loadModel(gptj, options.path)) {
        return .{ .state = null };
    }

    return .{ .state = gptj };
}

pub fn deinit(model: *Model) void {
    var ctx = @ptrCast(c.llmodel_model, model.state);
    c.llmodel_llama_destroy(ctx);
}
