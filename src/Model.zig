const std = @import("std");
const log = std.log;
const addNullByte = std.cstr.addNullByte;
const c = @import("c.zig");

state: ?*anyopaque,
response_fn: *const fn (response: []const u8) bool,

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
        .n_predict = 128,
        .top_k = 40,
        .top_p = 0.9,
        .temp = 0.1,
        .n_batch = 8,
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
    response_fn: *const fn (response: []const u8) bool,
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

    const callbacks = struct {
        pub var mm: *Model = undefined;
        pub var res: *std.ArrayList(u8) = undefined;

        fn promptFn(_: i32) callconv(.C) bool {
            return true;
        }
        fn recalculateFn(recalculate: bool) callconv(.C) bool {
            return recalculate;
        }
        fn respFn(_: i32, resp: [*c]const u8) callconv(.C) bool {
            res.appendSlice(std.mem.span(resp)) catch |err| {
                log.err("resp err: {s}", .{@errorName(err)});
            };
            return mm.response_fn(std.mem.span(resp));
        }
    };
    callbacks.res = &result;
    callbacks.mm = model;

    var prompt = try addNullByte(options.allocator, options.text);
    defer options.allocator.free(prompt);

    const PromptContext = [*c]c.llmodel_prompt_context;
    c.llmodel_prompt(
        ctx,
        prompt,
        callbacks.promptFn,
        callbacks.respFn,
        callbacks.recalculateFn,
        @intToPtr(PromptContext, @ptrToInt(&options.ctx)),
    );

    return try result.toOwnedSlice();
}

pub fn deinit(model: *Model) void {
    var ctx = @ptrCast(c.llmodel_model, model.state);
    c.llmodel_llama_destroy(ctx);
}
