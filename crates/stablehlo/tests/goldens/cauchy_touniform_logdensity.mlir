module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %4 = stablehlo.subtract %0, %3 : tensor<f32>
    %5 = stablehlo.subtract %2, %0 : tensor<f32>
    %6 = stablehlo.multiply %4, %5 : tensor<f32>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.compare GE, %6, %7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = stablehlo.constant dense<-1.1447298858494002> : tensor<f32>
    %10 = stablehlo.log %arg1 : tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = stablehlo.subtract %0, %arg0 : tensor<f32>
    %13 = stablehlo.divide %12, %arg1 : tensor<f32>
    %14 = stablehlo.multiply %13, %13 : tensor<f32>
    %15 = stablehlo.constant dense<1.0> : tensor<f32>
    %16 = stablehlo.add %15, %14 : tensor<f32>
    %17 = stablehlo.log %16 : tensor<f32>
    %18 = stablehlo.negate %17 : tensor<f32>
    %19 = stablehlo.add %9, %11 : tensor<f32>
    %20 = stablehlo.add %19, %18 : tensor<f32>
    %21 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %22 = stablehlo.negate %21 : tensor<f32>
    %23 = stablehlo.select %8, %20, %22 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %24 = stablehlo.subtract %2, %arg0 : tensor<f32>
    %25 = stablehlo.divide %24, %arg1 : tensor<f32>
    %26 = stablehlo.constant dense<1.0> : tensor<f32>
    %27 = stablehlo.atan2 %25, %26 : tensor<f32>
    %28 = stablehlo.constant dense<0.3183098861837907> : tensor<f32>
    %29 = stablehlo.multiply %28, %27 : tensor<f32>
    %30 = stablehlo.constant dense<0.5> : tensor<f32>
    %31 = stablehlo.add %30, %29 : tensor<f32>
    %32 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %33 = stablehlo.subtract %32, %arg0 : tensor<f32>
    %34 = stablehlo.divide %33, %arg1 : tensor<f32>
    %35 = stablehlo.constant dense<1.0> : tensor<f32>
    %36 = stablehlo.atan2 %34, %35 : tensor<f32>
    %37 = stablehlo.constant dense<0.3183098861837907> : tensor<f32>
    %38 = stablehlo.multiply %37, %36 : tensor<f32>
    %39 = stablehlo.constant dense<0.5> : tensor<f32>
    %40 = stablehlo.add %39, %38 : tensor<f32>
    %41 = stablehlo.subtract %31, %40 : tensor<f32>
    %42 = stablehlo.log %41 : tensor<f32>
    %43 = stablehlo.subtract %23, %42 : tensor<f32>
    return %43 : tensor<f32>
  }
}
