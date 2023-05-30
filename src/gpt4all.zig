const c = @import("c.zig");

pub fn load_mpt_model(fname: [*c]const u8, n_threads: i32) ?*anyopaque {
    // load the model
    var gptj = c.llmodel_mpt_create();

    c.llmodel_setThreadCount(gptj, n_threads);
    if (!c.llmodel_loadModel(gptj, fname)) {
        return null;
    }

    return gptj;
}

pub fn load_llama_model(fname: [*c]const u8, n_threads: i32) ?*anyopaque {
    // load the model
    var gptj = c.llmodel_llama_create();

    c.llmodel_setThreadCount(gptj, n_threads);
    if (!c.llmodel_loadModel(gptj, fname)) {
        return null;
    }

    return gptj;
}

pub fn load_gptj_model(fname: [*c]const u8, n_threads: i32) ?*anyopaque {
    // load the model
    var gptj = c.llmodel_gptj_create();

    c.llmodel_setThreadCount(gptj, n_threads);
    if (!c.llmodel_loadModel(gptj, fname)) {
        return null;
    }

    return gptj;
}
