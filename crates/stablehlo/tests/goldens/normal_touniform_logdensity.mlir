module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %arg1, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %7 = stablehlo.subtract %6, %arg0 : tensor<f32>
    %8 = stablehlo.constant dense<1.4142135623730951> : tensor<f32>
    %9 = stablehlo.multiply %arg1, %8 : tensor<f32>
    %10 = stablehlo.divide %7, %9 : tensor<f32>
    %11 = chlo.erf %10 : tensor<f32> -> tensor<f32>
    %12 = stablehlo.add %0, %11 : tensor<f32>
    %13 = stablehlo.constant dense<0.5> : tensor<f32>
    %14 = stablehlo.multiply %13, %12 : tensor<f32>
    %15 = stablehlo.subtract %0, %14 : tensor<f32>
    %16 = stablehlo.compare EQ, %15, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %17 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %18 = stablehlo.select %16, %0, %15 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %19 = stablehlo.log %18 : tensor<f32>
    %20 = stablehlo.select %16, %17, %19 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %21 = stablehlo.abs %20 : tensor<f32>
    %22 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %23 = stablehlo.compare LT, %21, %22 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %24 = stablehlo.constant dense<true> : tensor<i1>
    %25 = stablehlo.compare LE, %13, %22 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %26 = stablehlo.and %24, %25 : tensor<i1>
    %27 = stablehlo.select %26, %13, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %28 = stablehlo.log %arg1 : tensor<f32>
    %29 = stablehlo.negate %28 : tensor<f32>
    %30 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %31 = stablehlo.subtract %27, %arg0 : tensor<f32>
    %32 = stablehlo.divide %31, %arg1 : tensor<f32>
    %33 = stablehlo.constant dense<-0.5> : tensor<f32>
    %34 = stablehlo.multiply %32, %32 : tensor<f32>
    %35 = stablehlo.multiply %33, %34 : tensor<f32>
    %36 = stablehlo.add %29, %30 : tensor<f32>
    %37 = stablehlo.add %36, %35 : tensor<f32>
    %38 = stablehlo.negate %22 : tensor<f32>
    %39 = stablehlo.select %26, %37, %38 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %40 = stablehlo.subtract %39, %20 : tensor<f32>
    %41 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %42 = stablehlo.select %23, %40, %41 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %42, %5 : tensor<f32>, tensor<i32>
  }
}
