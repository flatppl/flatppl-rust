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
    %8 = stablehlo.divide %7, %arg1 : tensor<f32>
    %9 = stablehlo.atan2 %8, %0 : tensor<f32>
    %10 = stablehlo.constant dense<0.3183098861837907> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.constant dense<0.5> : tensor<f32>
    %13 = stablehlo.add %12, %11 : tensor<f32>
    %14 = stablehlo.subtract %0, %13 : tensor<f32>
    %15 = stablehlo.compare EQ, %14, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = stablehlo.select %15, %0, %14 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.log %17 : tensor<f32>
    %19 = stablehlo.select %15, %16, %18 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.abs %19 : tensor<f32>
    %21 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %22 = stablehlo.compare LT, %20, %21 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %23 = stablehlo.constant dense<true> : tensor<i1>
    %24 = stablehlo.compare LE, %12, %21 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %25 = stablehlo.and %23, %24 : tensor<i1>
    %26 = stablehlo.select %25, %12, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %27 = stablehlo.constant dense<-1.1447298858494002> : tensor<f32>
    %28 = stablehlo.log %arg1 : tensor<f32>
    %29 = stablehlo.negate %28 : tensor<f32>
    %30 = stablehlo.subtract %26, %arg0 : tensor<f32>
    %31 = stablehlo.divide %30, %arg1 : tensor<f32>
    %32 = stablehlo.multiply %31, %31 : tensor<f32>
    %33 = stablehlo.add %0, %32 : tensor<f32>
    %34 = stablehlo.log %33 : tensor<f32>
    %35 = stablehlo.negate %34 : tensor<f32>
    %36 = stablehlo.add %27, %29 : tensor<f32>
    %37 = stablehlo.add %36, %35 : tensor<f32>
    %38 = stablehlo.negate %21 : tensor<f32>
    %39 = stablehlo.select %25, %37, %38 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %40 = stablehlo.subtract %39, %19 : tensor<f32>
    %41 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %42 = stablehlo.select %22, %40, %41 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %42, %5 : tensor<f32>, tensor<i32>
  }
}
