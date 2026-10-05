package org.monarch.nn.backend;

final class TornadoBootstrap {

    static {
        // Runs as soon as the class is loaded by the JVM.
        if (Thread.currentThread().getContextClassLoader() == null) {
            Thread.currentThread().setContextClassLoader(
                    TornadoBootstrap.class.getClassLoader());
        }

        setIfAbsent("tornado.load.api.implementation",
                "uk.ac.manchester.tornado.runtime.tasks.TornadoTaskGraph");
        setIfAbsent("tornado.load.runtime.implementation",
                "uk.ac.manchester.tornado.runtime.core.TornadoCoreRuntime");
        setIfAbsent("tornado.load.tornado.implementation",
                "uk.ac.manchester.tornado.runtime.common.Tornado");
    }

    private TornadoBootstrap() {}

    /** Force class-init to run. Harmless to call multiple times. */
    static void init() { }

    private static void setIfAbsent(String key, String value) {
        if (System.getProperty(key) == null) {
            System.setProperty(key, value);
        }
    }
}