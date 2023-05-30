const std = @import("std");
const c = @import("c.zig");

state: ?*anyopaque,
response_fn: *const fn (
    token_id: i32,
    response: [*c]const u8,
) callconv(.C) bool,

const Model = @This();

const PredictOptions = struct {
    allocator: std.mem.Allocator,
    text: []const u8,
    ctx: c.llmodel_prompt_context = .{
        .logits = null,
        .logits_size = 0,
        .tokens = null,
        .tokens_size = 0,
        .n_past = 0,
        .n_ctx = 1024,
        .n_predict = 200,
        .top_k = 10,
        .top_p = 0.9,
        .temp = 0.96,
        .n_batch = 1,
        .repeat_penalty = 1.2,
        .repeat_last_n = 10,
        .context_erase = 0.55,
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
    response_fn: *const fn (
        token_id: i32,
        response: [*c]const u8,
    ) callconv(.C) bool,
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
        .response_fn = options.response_fn,
    };
}

pub fn predict(model: *Model, options: PredictOptions) ![]u8 {
    var result = std.ArrayList(u8).init(options.allocator);
    var ctx = @ptrCast(c.llmodel_model, model.state);

    //if (options.ctx.n_predict == 0) {
    //    options.ctx.n_predict = 99999999;
    //}

    const callbacks = struct {
        fn promptFn(_: i32) callconv(.C) bool {
            return true;
        }
        fn recalculateFn(recalculate: bool) callconv(.C) bool {
            return recalculate;
        }
    };

    c.llmodel_prompt(
        ctx,
        options.text.ptr,
        callbacks.promptFn,
        model.response_fn,
        callbacks.recalculateFn,
        @intToPtr(
            [*c]c.llmodel_prompt_context,
            @ptrToInt(&options.ctx),
        ),
    );

    return try result.toOwnedSlice();
}

pub fn deinit(model: *Model) void {
    var ctx = @ptrCast(c.llmodel_model, model.state);
    c.llmodel_llama_destroy(ctx);
}
